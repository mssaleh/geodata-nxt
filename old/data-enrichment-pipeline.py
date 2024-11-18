"""
Data enrichment pipeline for geodata.py
-------------------------------------

This module provides the main data enrichment pipeline that coordinates
all components to process and enrich location data.

Features:
- Coordinate validation and standardization
- Address parsing and normalization
- Geocoding service integration
- PostGIS compatibility
- Batch processing with progress tracking
"""

import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from typing import Dict, List, Optional, Union, Any, Iterator
from dataclasses import dataclass
import json
import logging
from shapely.geometry import Point
from shapely import wkb, wkt
import h3
from datetime import datetime
from pathlib import Path

# Import our components
from .geocoding import GeocodingService, GeocodingResponse
from .address import AddressStandardizer, AddressComponents
from .uae import UAEAddressHandler, UAEAddressComponents

logger = logging.getLogger(__name__)

@dataclass
class EnrichmentConfig:
    """Configuration for data enrichment process."""
    input_file: Path
    output_file: Path
    batch_size: int = 100
    h3_resolution: int = 9
    validate_coordinates: bool = True
    enrich_addresses: bool = True
    standardize_arabic: bool = True
    add_uae_parsing: bool = True
    geometry_format: str = 'wkb'  # or 'wkt'
    cache_responses: bool = True
    error_handling: str = 'warn'  # 'strict', 'warn', or 'ignore'
    progress_log_interval: int = 1000

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
    
    def __init__(self, config: EnrichmentConfig):
        self.config = config
        self._initialize_components()
        self._initialize_stats()
    
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        self.geocoding_service = GeocodingService()
        self.address_standardizer = AddressStandardizer()
        self.uae_handler = UAEAddressHandler()
        
        logger.info("Initialized enrichment components")
    
    def _initialize_stats(self) -> None:
        """Initialize processing statistics."""
        self.stats = ProcessingStats()
    
    def process_file(self) -> None:
        """Process the entire input file and generate enriched output."""
        try:
            self.stats.start_time = datetime.now()
            logger.info(f"Starting processing of {self.config.input_file}")
            
            # Read input file with Dask
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
            
            self.stats.total_records = len(df)
            logger.info(f"Found {self.stats.total_records} records to process")
            
            # Process in batches
            with ProgressBar():
                enriched_data = []
                
                for batch in self._get_batches(df):
                    processed_batch = self._process_batch(batch)
                    enriched_data.extend(processed_batch)
                    
                    # Log progress
                    if len(enriched_data) % self.config.progress_log_interval == 0:
                        self._log_progress()
            
            # Convert to DataFrame and save
            output_df = pd.DataFrame(enriched_data)
            self._save_output(output_df)
            
            self.stats.end_time = datetime.now()
            self._log_final_stats()
            
        except Exception as e:
            logger.error(f"Critical error during file processing: {str(e)}", exc_info=True)
            raise
    
    def _get_batches(self, df: dd.DataFrame) -> Iterator[pd.DataFrame]:
        """Generate batches from DataFrame."""
        total_partitions = df.npartitions
        for i in range(total_partitions):
            partition = df.get_partition(i).compute()
            for start_idx in range(0, len(partition), self.config.batch_size):
                yield partition.iloc[start_idx:start_idx + self.config.batch_size]
    
    def _process_batch(self, batch: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a batch of records."""
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
                logger.error(f"Error processing record: {str(e)}")
                if self.config.error_handling == 'strict':
                    raise
                    
            self.stats.processed_records += 1
            
        return enriched_records
    
    def _process_record(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Process a single record."""
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
            address_data = self._process_address_data(row)
            if address_data:
                enriched.update(address_data)
            
            # Add additional metadata
            enriched['processing_timestamp'] = datetime.now().isoformat()
            
            return enriched
            
        except Exception as e:
            logger.error(f"Error in record processing: {str(e)}")
            return None
    
    def _extract_coordinates(self, row: pd.Series) -> Optional[Dict[str, float]]:
        """Extract and validate coordinates from row."""
        try:
            if isinstance(row.get('coordinates'), str):
                coords = json.loads(row['coordinates'].replace("'", '"'))
            else:
                coords = row.get('coordinates', {})
            
            lat = float(coords.get('latitude', 0))
            lng = float(coords.get('longitude', 0))
            
            # Validate coordinates
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                logger.warning(f"Invalid coordinates: {lat}, {lng}")
                return None
            
            return {'lat': lat, 'lng': lng}
            
        except Exception as e:
            logger.error(f"Error extracting coordinates: {str(e)}")
            return None
    
    def _generate_geometries(self, lat: float, lng: float) -> Dict[str, str]:
        """Generate geometry representations."""
        point = Point(lng, lat)  # Note: PostGIS uses (longitude, latitude) order
        
        result = {
            'h3_index': h3.geo_to_h3(lat, lng, self.config.h3_resolution),
            'geom_wkb': wkb.dumps(point, hex=True, srid=4326),
            'geom_wkt': wkt.dumps(point)
        }
        
        # Add parent H3 indices for hierarchical queries
        parent_indices = []
        for res in range(self.config.h3_resolution - 1, 4, -1):  # Up to resolution 4
            parent_indices.append(h3.h3_to_parent(result['h3_index'], res))
        result['h3_hierarchy'] = json.dumps(parent_indices)
        
        return result
    
    def _process_address_data(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Process and normalize address data."""
        try:
            address_data = {}
            
            # Parse detailed address if available
            if 'detailed_address' in row:
                if isinstance(row['detailed_address'], str):
                    detailed_address = json.loads(row['detailed_address'].replace("'", '"'))
                else:
                    detailed_address = row['detailed_address']
                
                # Standardize address components
                std_components = self.address_standardizer.standardize_address(detailed_address)
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
                if std_components.country_code == 'AE':
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
            logger.error(f"Error processing address data: {str(e)}")
            return None
    
    def _save_output(self, df: pd.DataFrame) -> None:
        """Save processed data to output file."""
        logger.info(f"Saving {len(df)} records to {self.config.output_file}")
        df.to_csv(self.config.output_file, index=False, encoding='utf-8')
    
    def _log_progress(self) -> None:
        """Log processing progress."""
        elapsed = datetime.now() - self.stats.start_time
        rate = self.stats.processed_records / elapsed.total_seconds()
        
        logger.info(
            f"Processed {self.stats.processed_records}/{self.stats.total_records} records "
            f"({(self.stats.processed_records/self.stats.total_records)*100:.1f}%) "
            f"Rate: {rate:.1f} records/second"
        )
    
    def _log_final_stats(self) -> None:
        """Log final processing statistics."""
        elapsed = self.stats.end_time - self.stats.start_time
        
        logger.info(
            f"\nProcessing completed:\n"
            f"- Total records: {self.stats.total_records}\n"
            f"- Successful: {self.stats.successful_records}\n"
            f"- Failed: {self.stats.failed_records}\n"
            f"- Invalid coordinates: {self.stats.invalid_coordinates}\n"
            f"- Geocoding failures: {self.stats.geocoding_failures}\n"
            f"- Address parse failures: {self.stats.address_parse_failures}\n"
            f"- Processing time: {elapsed.total_seconds():.1f} seconds\n"
            f"- Average rate: {self.stats.processed_records/elapsed.total_seconds():.1f} records/second"
        )
