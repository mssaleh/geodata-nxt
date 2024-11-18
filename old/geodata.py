"""
Geospatial Data Enrichment Script
--------------------------------

This script enriches location data with additional geospatial information and validates
existing geographical data. It handles large datasets efficiently using Dask and 
provides robust error handling and logging.

Features:
- H3 geospatial indexing for proximity queries
- ISO 3166-2 code validation and normalization
- Administrative boundary validation
- Timezone verification
- PostGIS-compatible geometry generation
- Batch processing with progress tracking
- Comprehensive logging
- Memory-efficient processing using Dask

Usage:
    python geodata.py --input places.csv --output enriched_places.csv [options]

Requirements:
    - Python 3.7+
    - Required packages: See requirements.txt
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import h3  # type: ignore
import numpy as np
import pandas as pd
import pycountry  # type: ignore
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from shapely.geometry import Point
from thefuzz import fuzz  # type: ignore
from timezonefinder import TimezoneFinder  # type: ignore

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(
            f"geodata_enrichment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Custom exceptions
class GeodataError(Exception):
    """Base exception for geodata processing errors."""
    pass

class DataValidationError(GeodataError):
    """Raised when data validation fails."""
    pass

class CoordinateParsingError(GeodataError):
    """Raised when coordinate parsing fails."""
    pass

@dataclass
class GeodataConfig:
    """Configuration for geospatial data enrichment.
    
    Attributes:
        input_file (Path): Path to input CSV file
        output_file (Path): Path to output CSV file
        batch_size (int): Number of records to process in each batch
        h3_resolution (int): Resolution for H3 grid cells (7-12)
        validate_timezone (bool): Whether to validate timezone data
        enrich_boundaries (bool): Whether to enrich administrative boundaries
        error_handling (str): How to handle errors ('strict', 'warn', or 'ignore')
    """
    input_file: Path
    output_file: Path
    batch_size: int = 100
    h3_resolution: int = 9
    validate_timezone: bool = True
    enrich_boundaries: bool = True
    error_handling: str = 'warn'
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        if not 7 <= self.h3_resolution <= 12:
            raise ValueError("H3 resolution must be between 7 and 12")
        
        if self.error_handling not in ('strict', 'warn', 'ignore'):
            raise ValueError("Invalid error_handling value")
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

class LocationDataEnricher:
    """Enriches location data with additional geospatial information.
    
    This class handles the processing of location data, including validation,
    enrichment, and standardization of geographical information.
    """
    
    def __init__(self, config: GeodataConfig):
        """Initialize the enricher with configuration.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self.tf = TimezoneFinder()
        self.admin_cache: Dict[str, Dict] = {}
        self.error_count = 0
        self.warning_count = 0
        
        logger.info(f"Initializing LocationDataEnricher with config: {config}")
    
    def _handle_error(self, error: Exception, context: str) -> None:
        """Handle errors according to configuration.
        
        Args:
            error: The exception that occurred
            context: Description of where the error occurred
        
        Raises:
            GeodataError: If error_handling is 'strict'
        """
        self.error_count += 1
        
        if self.config.error_handling == 'strict':
            raise GeodataError(f"Error in {context}: {str(error)}")
        elif self.config.error_handling == 'warn':
            logger.warning(f"Error in {context}: {str(error)}")
        # 'ignore' - do nothing
    
    def _parse_coordinates(self, coord_data: Union[Dict, str, None]) -> Tuple[float, float]:
        """Extract latitude and longitude from coordinates field.
        
        Args:
            coord_data: Coordinate data in dictionary or string format
            
        Returns:
            Tuple of (latitude, longitude)
            
        Raises:
            CoordinateParsingError: If coordinates cannot be parsed
        """
        try:
            if coord_data is None:
                raise ValueError("Coordinate data is None")
                
            if isinstance(coord_data, dict):
                lat = float(coord_data.get('latitude', 0))
                lng = float(coord_data.get('longitude', 0))
            elif isinstance(coord_data, str):
                coord_dict = json.loads(coord_data.replace("'", '"'))
                lat = float(coord_dict.get('latitude', 0))
                lng = float(coord_dict.get('longitude', 0))
            else:
                raise ValueError(f"Unsupported coordinate data type: {type(coord_data)}")
            
            # Basic coordinate validation
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                raise ValueError(f"Invalid coordinate values: lat={lat}, lng={lng}")
                
            return lat, lng
            
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            raise CoordinateParsingError(f"Failed to parse coordinates: {str(e)}")
    
    def _generate_h3_index(self, lat: float, lng: float) -> str:
        """Generate H3 geohash index for coordinates.
        
        Args:
            lat: Latitude
            lng: Longitude
            
        Returns:
            H3 index as string
        """
        try:
            return h3.geo_to_h3(lat, lng, self.config.h3_resolution)
        except ValueError as e:
            self._handle_error(e, "H3 index generation")
            return ""
    
    def _validate_timezone(self, lat: float, lng: float, current_tz: str) -> str:
        """Validate and potentially correct timezone information.
        
        Args:
            lat: Latitude
            lng: Longitude
            current_tz: Current timezone string
            
        Returns:
            Validated timezone string
        """
        if not self.config.validate_timezone:
            return current_tz
            
        try:
            computed_tz = self.tf.timezone_at(lat=lat, lng=lng)
            if computed_tz and computed_tz != current_tz:
                self.warning_count += 1
                logger.warning(
                    f"Timezone mismatch at ({lat}, {lng}) - "
                    f"Stored: {current_tz}, Computed: {computed_tz}"
                )
                return computed_tz
        except Exception as e:
            self._handle_error(e, "timezone validation")
            
        return current_tz
    
    def _normalize_iso3166_2(self, country_code: str, state: str) -> str:
        """Normalize ISO 3166-2 codes using pycountry.
        
        Args:
            country_code: Two-letter country code
            state: State/province name
            
        Returns:
            Normalized ISO 3166-2 code
        """
        cache_key = f"{country_code}:{state}"
        if cache_key in self.admin_cache:
            return self.admin_cache[cache_key].get('iso3166_2', '')
            
        if not country_code or not state:
            return ""
            
        try:
            country = pycountry.countries.get(alpha_2=country_code.upper())
            if not country:
                return ""
                
            subdivisions = list(pycountry.subdivisions.get(country_code=country.alpha_2))
            if not subdivisions:
                return ""
            
            # Find best matching subdivision using fuzzy matching
            best_match = max(
                subdivisions,
                key=lambda x: fuzz.ratio(state.lower(), x.name.lower())
            )
            
            match_ratio = fuzz.ratio(state.lower(), best_match.name.lower())
            if match_ratio > 80:  # Confidence threshold
                result = best_match.code
                # Cache the result
                self.admin_cache[cache_key] = {'iso3166_2': result}
                return result
                
        except Exception as e:
            self._handle_error(e, "ISO 3166-2 normalization")
            
        return f"{country_code.upper()}-{state.upper()}"
    
    def _extract_admin_boundaries(self, detailed_address: Dict) -> Dict[str, str]:
        """Extract and validate administrative boundaries.
        
        Args:
            detailed_address: Dictionary containing address components
            
        Returns:
            Dictionary of validated boundary information
        """
        try:
            result = {
                'country_code': detailed_address.get('country_code', '').upper(),
                'state': detailed_address.get('state', ''),
                'city': detailed_address.get('city', ''),
                'ward': detailed_address.get('ward', ''),
                'postal_code': detailed_address.get('postal_code', ''),
                'street': detailed_address.get('street', '')
            }
            
            # Normalize ISO 3166-2 code if we have both country and state
            if result['country_code'] and result['state']:
                result['iso3166_2'] = self._normalize_iso3166_2(
                    result['country_code'],
                    result['state']
                )
            else:
                result['iso3166_2'] = ''
            
            return result
            
        except Exception as e:
            self._handle_error(e, "admin boundary extraction")
            return {}
    
    def _prepare_for_database(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Format location data for database storage.
        
        Args:
            row: Dictionary containing location data
            
        Returns:
            Dictionary of formatted data ready for database storage
        """
        try:
            # Parse coordinates
            lat, lng = self._parse_coordinates(row.get('coordinates'))
            
            # Generate point geometry for PostGIS
            point_wkt = f"POINT({lng} {lat})"
            
            # Extract and validate address components
            address_data = self._extract_admin_boundaries(
                row.get('detailed_address', {})
            )
            
            # Generate H3 index
            h3_index = self._generate_h3_index(lat, lng)
            
            # Validate timezone
            timezone = self._validate_timezone(
                lat,
                lng,
                row.get('time_zone', '')
            )
            
            # Construct enriched record
            enriched = {
                'place_id': row.get('place_id', ''),
                'name': row.get('name', ''),
                'latitude': lat,
                'longitude': lng,
                'geometry': point_wkt,
                'h3_index': h3_index,
                'plus_code': row.get('plus_code', ''),
                'timezone': timezone,
                'original_data': json.dumps(row)
            }
            
            # Add address data
            enriched.update(address_data)
            
            return enriched
            
        except Exception as e:
            self._handle_error(e, "database record preparation")
            return {}
    
    def process_file(self) -> None:
        """Process the input file and enrich with geospatial data."""
        start_time = time.time()
        
        try:
            # Read input file with Dask
            df = dd.read_csv(
                self.config.input_file,
                assume_missing=True,
                dtype={
                    'place_id': 'object',
                    'coordinates': 'object',
                    'detailed_address': 'object',
                    'time_zone': 'object'
                },
                blocksize='64MB'  # Optimize for memory usage
            )
            
            total_rows = len(df)
            logger.info(f"Processing {total_rows} rows from {self.config.input_file}")
            
            # Process in batches
            enriched_data = []
            
            with ProgressBar():
                for i in range(0, total_rows, self.config.batch_size):
                    batch = df.iloc[i:i + self.config.batch_size].compute()
                    
                    for _, row in batch.iterrows():
                        # Convert row to dict and parse JSON fields
                        row_dict = row.to_dict()
                        for field in ['coordinates', 'detailed_address']:
                            if isinstance(row_dict.get(field), str):
                                try:
                                    row_dict[field] = json.loads(
                                        row_dict[field].replace("'", '"')
                                    )
                                except (json.JSONDecodeError, AttributeError):
                                    row_dict[field] = {}
                        
                        # Prepare enriched record
                        enriched_record = self._prepare_for_database(row_dict)
                        if enriched_record:
                            enriched_data.append(enriched_record)
                    
                    # Log progress
                    if (i + 1) % 1000 == 0:
                        logger.info(f"Processed {i + 1} records...")
            
            # Convert to DataFrame for efficient CSV writing
            output_df = pd.DataFrame(enriched_data)
            
            # Save enriched data
            output_df.to_csv(
                self.config.output_file,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                encoding='utf-8'
            )
            
            # Log summary statistics
            processing_time = time.time() - start_time
            logger.info(
                f"Processing complete:\n"
                f"- Total records: {total_rows}\n"
                f"- Successfully processed: {len(enriched_data)}\n"
                f"- Errors: {self.error_count}\n"
                f"- Warnings: {self.warning_count}\n"
                f"- Processing time: {processing_time:.2f} seconds\n"
                f"- Output file: {self.config.output_file}"
            )
            
        except Exception as e:
            logger.error(f"Critical error processing file: {str(e)}", exc_info=True)
            raise

def main() -> None:
    """Main entry point for the script.
    
    Parses command line arguments, validates inputs, initializes the data enricher,
    and handles the overall execution flow and error reporting.
    
    Command line arguments:
        --input: Path to input CSV file
        --output: Path to save enriched CSV file
        --batch-size: Number of records to process in each batch
        --h3-resolution: Resolution for H3 grid cells (7-12)
        --error-handling: How to handle errors ('strict', 'warn', or 'ignore')
        --skip-timezone-validation: Skip timezone validation step
        --skip-boundary-enrichment: Skip administrative boundary enrichment
        --log-level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    parser = argparse.ArgumentParser(
        description="Enrich and validate location data from Google Places",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file containing place data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save enriched CSV file'
    )
    
    # Optional processing parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of records to process in each batch'
    )
    
    parser.add_argument(
        '--h3-resolution',
        type=int,
        default=9,
        choices=range(7, 13),
        help='Resolution for H3 grid cells (7-12)'
    )
    
    parser.add_argument(
        '--error-handling',
        type=str,
        default='warn',
        choices=['strict', 'warn', 'ignore'],
        help='How to handle processing errors'
    )
    
    # Feature flags
    parser.add_argument(
        '--skip-timezone-validation',
        action='store_true',
        help='Skip timezone validation step'
    )
    
    parser.add_argument(
        '--skip-boundary-enrichment',
        action='store_true',
        help='Skip administrative boundary enrichment'
    )
    
    # Logging configuration
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        # Validate output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration
        config = GeodataConfig(
            input_file=input_path,
            output_file=output_path,
            batch_size=args.batch_size,
            h3_resolution=args.h3_resolution,
            validate_timezone=not args.skip_timezone_validation,
            enrich_boundaries=not args.skip_boundary_enrichment,
            error_handling=args.error_handling
        )
        
        # Log startup information
        logger.info(
            f"Starting geospatial data enrichment:\n"
            f"- Input file: {config.input_file}\n"
            f"- Output file: {config.output_file}\n"
            f"- Batch size: {config.batch_size}\n"
            f"- H3 resolution: {config.h3_resolution}\n"
            f"- Error handling: {config.error_handling}\n"
            f"- Timezone validation: {'enabled' if config.validate_timezone else 'disabled'}\n"
            f"- Boundary enrichment: {'enabled' if config.enrich_boundaries else 'disabled'}"
        )
        
        # Initialize and run enricher
        start_time = time.time()
        enricher = LocationDataEnricher(config)
        enricher.process_file()
        
        # Log completion statistics
        processing_time = time.time() - start_time
        logger.info(
            f"Processing completed successfully:\n"
            f"- Total processing time: {processing_time:.2f} seconds\n"
            f"- Error count: {enricher.error_count}\n"
            f"- Warning count: {enricher.warning_count}"
        )
        
    except Exception as e:
        logger.error(f"Critical error during execution: {str(e)}", exc_info=True)
        sys.exit(1)
    
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(2)
    
    sys.exit(0)

if __name__ == "__main__":
    main()