"""
Core data enrichment module for geodata processing.

This module provides the main data enrichment pipeline that coordinates
all components to process and enrich location data. It handles:
- Coordinate validation and standardization
- Address parsing and normalization
- Geocoding service integration
- PostGIS compatibility
- Batch processing with progress tracking
- Memory-efficient processing using Dask
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import dask.dataframe as dd
import h3
import pandas as pd
from dask.diagnostics import ProgressBar
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from shapely.geometry import Point
from shapely import wkb, wkt

from ..utils.config import EnrichmentConfig
from ..utils.exceptions import (
    CoordinateError,
    EnrichmentError,
    GeodataError,
    ValidationError
)
from .geocoding import GeocodingService, GeocodingResponse
from .address import AddressStandardizer, AddressComponents
from .uae import UAEAddressHandler, UAEAddressComponents

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Statistics for the enrichment process."""
    total_records: int = 0
    processed_records: int = 0
    successful_records: int = 0
    failed_records: int = 0
    invalid_coordinates: int = 0
    geocoding_failures: int = 0
    address_parse_failures: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def get_summary(self) -> Dict[str, Any]:
        """Get a dictionary summary of processing statistics."""
        elapsed = (self.end_time - self.start_time) if self.end_time else None
        return {
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "successful_records": self.successful_records,
            "failed_records": self.failed_records,
            "invalid_coordinates": self.invalid_coordinates,
            "geocoding_failures": self.geocoding_failures,
            "address_parse_failures": self.address_parse_failures,
            "processing_time": elapsed.total_seconds() if elapsed else None,
            "success_rate": (
                (self.successful_records / self.total_records * 100)
                if self.total_records else 0
            ),
        }

class LocationDataEnricher:
    """
    Main pipeline for enriching location data.
    
    Coordinates all components to process and enrich location data with:
    - Standardized coordinates
    - Normalized addresses
    - Administrative boundaries
    - PostGIS-compatible geometries
    - H3 indices
    """
    
    def __init__(self, config: EnrichmentConfig) -> None:
        """
        Initialize the enricher with configuration.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self.stats = ProcessingStats()
        self._initialize_components()
        logger.info("LocationDataEnricher initialized with config: %s", config)

    def _initialize_components(self) -> None:
        """Initialize all required components."""
        try:
            self.geocoding_service = GeocodingService()
            self.address_standardizer = AddressStandardizer()
            self.uae_handler = UAEAddressHandler()
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize components: %s", str(e))
            raise EnrichmentError("Component initialization failed") from e

    def process_file(self) -> None:
        """
        Process the entire input file and generate enriched output.
        
        Raises:
            EnrichmentError: If processing fails
            FileNotFoundError: If input file doesn't exist
            ValidationError: If input data is invalid
        """
        try:
            self.stats.start_time = datetime.now()
            logger.info("Starting processing of %s", self.config.input_file)

            # Validate input file
            if not self.config.input_file.exists():
                raise FileNotFoundError(f"Input file not found: {self.config.input_file}")

            # Read input file with Dask
            df = self._read_input_file()
            self.stats.total_records = len(df)
            logger.info("Found %d records to process", self.stats.total_records)

            # Process in batches
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
            ) as progress:
                enriched_data = []
                task = progress.add_task(
                    "Enriching data...",
                    total=self.stats.total_records
                )

                for batch in self._get_batches(df):
                    processed_batch = self._process_batch(batch)
                    enriched_data.extend(processed_batch)
                    
                    # Update progress
                    progress.advance(task, len(batch))
                    self._log_progress()

            # Convert to DataFrame and save
            self._save_output(enriched_data)
            
            self.stats.end_time = datetime.now()
            self._log_final_stats()

        except Exception as e:
            logger.error("Critical error during file processing: %s", str(e), exc_info=True)
            raise EnrichmentError("File processing failed") from e

    def _read_input_file(self) -> dd.DataFrame:
        """
        Read and validate the input file.
        
        Returns:
            Dask DataFrame containing input data
            
        Raises:
            ValidationError: If input data is invalid
        """
        try:
            df = dd.read_csv(
                self.config.input_file,
                assume_missing=True,
                dtype={
                    'place_id': 'object',
                    'coordinates': 'object',
                    'detailed_address': 'object'
                },
                blocksize='64MB'
            )

            # Validate required columns
            required_columns = {'place_id', 'coordinates', 'detailed_address'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValidationError(
                    f"Missing required columns: {', '.join(missing_columns)}"
                )

            return df

        except Exception as e:
            logger.error("Error reading input file: %s", str(e))
            raise ValidationError("Failed to read input file") from e

    def _get_batches(self, df: dd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        Generate batches from DataFrame.
        
        Args:
            df: Input Dask DataFrame
            
        Yields:
            Pandas DataFrame containing batch of records
        """
        total_partitions = df.npartitions
        for i in range(total_partitions):
            partition = df.get_partition(i).compute()
            for start_idx in range(0, len(partition), self.config.batch_size):
                yield partition.iloc[start_idx:start_idx + self.config.batch_size]

    def _process_batch(self, batch: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a batch of records.
        
        Args:
            batch: Batch of records to process
            
        Returns:
            List of processed records
        """
        enriched_records = []
        
        for _, row in batch.iterrows():
            try:
                enriched_record = self._process_record(row)
                if enriched_record:
                    enriched_records.append(enriched_record)
                    self.stats.successful_records += 1
                else:
                    self.stats.failed_records += 1
                    
            except Exception as e:
                self.stats.failed_records += 1
                logger.error("Error processing record: %s", str(e))
                if self.config.error_handling == 'strict':
                    raise
                    
            self.stats.processed_records += 1
            
        return enriched_records

    def _process_record(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Process a single record.
        
        Args:
            row: Single record to process
            
        Returns:
            Processed record or None if processing failed
        """
        try:
            # Extract and validate coordinates
            coords = self._extract_coordinates(row)
            if not coords:
                self.stats.invalid_coordinates += 1
                return None
            
            # Create base enriched record
            enriched = {
                'place_id': row.get('place_id', ''),
                'name': row.get('name', ''),
                'original_coordinates': row.get('coordinates', ''),
                'latitude': coords['lat'],
                'longitude': coords['lng']
            }
            
            # Add geometries
            geometry_data = self._generate_geometries(coords['lat'], coords['lng'])
            enriched.update(geometry_data)
            
            # Process address data
            if self.config.enrich_addresses:
                address_data = self._process_address_data(row)
                if address_data:
                    enriched.update(address_data)
            
            # Add additional metadata
            enriched['processing_timestamp'] = datetime.now().isoformat()
            
            return enriched
            
        except Exception as e:
            logger.error("Error in record processing: %s", str(e))
            return None

    def _extract_coordinates(self, row: pd.Series) -> Optional[Dict[str, float]]:
        """
        Extract and validate coordinates from row.
        
        Args:
            row: Row containing coordinate data
            
        Returns:
            Dictionary with validated coordinates or None if invalid
            
        Raises:
            CoordinateError: If coordinates cannot be extracted or are invalid
        """
        try:
            if isinstance(row.get('coordinates'), str):
                coords = json.loads(row['coordinates'].replace("'", '"'))
            else:
                coords = row.get('coordinates', {})
            
            lat = float(coords.get('latitude', 0))
            lng = float(coords.get('longitude', 0))
            
            # Validate coordinates if enabled
            if self.config.validate_coordinates:
                if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                    raise CoordinateError(f"Invalid coordinate values: lat={lat}, lng={lng}")
            
            return {'lat': lat, 'lng': lng}
            
        except Exception as e:
            logger.error("Error extracting coordinates: %s", str(e))
            return None

    def _generate_geometries(self, lat: float, lng: float) -> Dict[str, str]:
        """
        Generate geometry representations.
        
        Args:
            lat: Latitude
            lng: Longitude
            
        Returns:
            Dictionary containing geometry representations
        """
        point = Point(lng, lat)  # Note: PostGIS uses (longitude, latitude) order
        result: Dict[str, str] = {}

        if self.config.geometry_format in ('wkb', 'both'):
            result['geom_wkb'] = wkb.dumps(point, hex=True, srid=4326)
            
        if self.config.geometry_format in ('wkt', 'both'):
            result['geom_wkt'] = wkt.dumps(point)

        # Add H3 index
        h3_index = h3.geo_to_h3(lat, lng, self.config.h3_resolution)
        result['h3_index'] = h3_index

        # Add parent H3 indices for hierarchical queries
        parent_indices = []
        for res in range(self.config.h3_resolution - 1, 4, -1):
            parent_indices.append(h3.h3_to_parent(h3_index, res))
        result['h3_hierarchy'] = json.dumps(parent_indices)

        return result

    def _process_address_data(
        self, 
        row: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """
        Process and normalize address data.
        
        Args:
            row: Row containing address data
            
        Returns:
            Dictionary of processed address data or None if processing failed
        """
        try:
            address_data = {}
            
            # Parse detailed address if available
            if 'detailed_address' in row:
                if isinstance(row['detailed_address'], str):
                    detailed_address = json.loads(
                        row['detailed_address'].replace("'", '"')
                    )
                else:
                    detailed_address = row['detailed_address']
                
                # Standardize address components
                std_components = self.address_standardizer.standardize_address(
                    detailed_address
                )
                address_data.update({
                    'country': std_components.country,
                    'country_code': std_components.country_code,
                    'state': std_components.state,
                    'city': std_components.city,
                    'district': std_components.district,
                    'postal_code': std_components.postal_code,
                    'formatted_address': std_components.formatted_address,
                    'iso3166_2': std_components.iso3166_2
                })
                
                # Special handling for UAE addresses
                if (self.config.add_uae_parsing and 
                    std_components.country_code == 'AE'):
                    uae_components = self.uae_handler.parse_uae_address(
                        std_components.formatted_address
                    )
                    address_data.update({
                        'uae_emirate': uae_components.emirate,
                        'uae_area': uae_components.area,
                        'uae_building': uae_components.building,
                        'uae_po_box': uae_components.po_box,
                        'makani_number': uae_components.makani
                    })
            
            return address_data
            
        except Exception as e:
            self.stats.address_parse_failures += 1
            logger.error("Error processing address data: %s", str(e))
            return None

    def _save_output(self, enriched_data: List[Dict[str, Any]]) -> None:
        """
        Save processed data to output file.
        
        Args:
            enriched_data: List of processed records to save
        """
        logger.info(
            "Saving %d records to %s",
            len(enriched_data),
            self.config.output_file
        )
        
        df = pd.DataFrame(enriched_data)
        df.to_csv(self.config.output_file, index=False, encoding='utf-8')

    def _log_progress(self) -> None:
        """Log processing progress."""
        if (self.stats.processed_records % 
            self.config.progress_log_interval == 0):
            elapsed = datetime.now() - self.stats.start_time
            rate = self.stats.processed_records / elapsed.total_seconds()
            
            logger.info(
                "Processed %d/%d records (%.1f%%) Rate: %.1f records/second",
                self.stats.processed_records,
                self.stats.total_records,
                (self.stats.processed_records/self.stats.total_records)*100,
                rate
            )

    def _log_final_stats(self) -> None:
        """Log final processing statistics."""
        stats = self.stats.get_summary()
        elapsed = self.stats.end_time - self.stats.start_time
        
        logger.info(
            "\nProcessing completed:\n"
            "- Total records: %d\n"
            "- Successful: %d\n"
            "- Failed: %d\n"
            "- Invalid coordinates: %d\n"
            "- Geocoding failures: %d\n"
            "- Address parse failures: %d\n"
            "- Processing time: %.1f seconds\n"
            "- Average rate: %.1f records/second\n"
            "- Success rate: %.1f%%",
            stats["total_records"],
            stats["successful_records"],
            stats["failed_records"],
            stats["invalid_coordinates"],
            stats["geocoding_failures"],
            stats["address_parse_failures"],
            elapsed.total_seconds(),
            stats["total_records"] / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0,
            stats["success_rate"]
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary containing current processing statistics
        """
        return self.stats.get_summary()

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'LocationDataEnricher':
        """
        Create enricher instance from YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configured LocationDataEnricher instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        from ..utils.config import EnrichmentConfig
        config = EnrichmentConfig.from_yaml(config_path)
        return cls(config)

    def __repr__(self) -> str:
        """Return string representation of enricher."""
        return (
            f"LocationDataEnricher("
            f"config={self.config}, "
            f"processed={self.stats.processed_records}, "
            f"successful={self.stats.successful_records}, "
            f"failed={self.stats.failed_records})"
        )

class ParallelLocationDataEnricher(LocationDataEnricher):
    """
    Extension of LocationDataEnricher that optimizes for parallel processing.
    
    This class overrides certain methods to better utilize Dask's parallel
    processing capabilities for larger datasets.
    """
    
    def _process_batch(self, batch: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a batch of records in parallel.
        
        Args:
            batch: Batch of records to process
            
        Returns:
            List of processed records
        """
        # Convert batch to dask dataframe for parallel processing
        ddf = dd.from_pandas(batch, npartitions=max(1, len(batch) // 1000))
        
        # Apply processing in parallel
        result = ddf.map_partitions(
            self._process_partition,
            meta=object
        ).compute()
        
        # Flatten results
        enriched_records = [
            record for partition in result 
            for record in partition 
            if record is not None
        ]
        
        # Update statistics
        self.stats.successful_records += len(enriched_records)
        self.stats.failed_records += len(batch) - len(enriched_records)
        self.stats.processed_records += len(batch)
        
        return enriched_records

    def _process_partition(self, partition: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a partition of records.
        
        Args:
            partition: Partition of records to process
            
        Returns:
            List of processed records
        """
        enriched_records = []
        
        for _, row in partition.iterrows():
            try:
                enriched_record = self._process_record(row)
                if enriched_record:
                    enriched_records.append(enriched_record)
            except Exception as e:
                logger.error(
                    "Error processing record in partition: %s", 
                    str(e)
                )
                if self.config.error_handling == 'strict':
                    raise
        
        return enriched_records

    def _save_output(self, enriched_data: List[Dict[str, Any]]) -> None:
        """
        Save processed data to output file using parallel writing.
        
        Args:
            enriched_data: List of processed records to save
        """
        logger.info(
            "Saving %d records to %s",
            len(enriched_data),
            self.config.output_file
        )
        
        # Convert to dask dataframe for parallel writing
        df = dd.from_pandas(
            pd.DataFrame(enriched_data),
            npartitions=max(1, len(enriched_data) // 10000)
        )
        
        # Save with progress bar
        with ProgressBar():
            df.to_csv(
                self.config.output_file,
                single_file=True,
                index=False,
                encoding='utf-8'
            )

def create_enricher(
    config: EnrichmentConfig,
    parallel: bool = True
) -> LocationDataEnricher:
    """
    Factory function to create appropriate enricher instance.
    
    Args:
        config: Enrichment configuration
        parallel: Whether to use parallel processing
        
    Returns:
        Configured enricher instance
    """
    if parallel:
        return ParallelLocationDataEnricher(config)
    return LocationDataEnricher(config)
