"""Coordinate validation and standardization module."""

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any
import h3
from shapely.geometry import Point, Polygon
import numpy as np

from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

@dataclass
class CoordinateValidationConfig:
    """Configuration for coordinate validation."""
    lat_min: float = -90.0
    lat_max: float = 90.0
    lng_min: float = -180.0
    lng_max: float = 180.0
    require_precision: int = 6
    check_land_area: bool = False
    validate_h3: bool = True
    h3_resolution: int = 9
    strict_mode: bool = False

@dataclass
class ValidationResult:
    """Results of coordinate validation."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    normalized: Optional[Dict[str, float]] = None

class CoordinateValidator:
    """Validates and normalizes geographic coordinates."""
    
    def __init__(self, config: Optional[CoordinateValidationConfig] = None):
        """Initialize validator with optional configuration."""
        self.config = config or CoordinateValidationConfig()
        self._validation_stats = {
            'total_validated': 0,
            'total_valid': 0,
            'total_invalid': 0,
            'error_types': {}
        }
    
    def validate(
        self,
        coordinates: Union[Dict[str, Any], str, Tuple[float, float]]
    ) -> ValidationResult:
        """Validate and normalize coordinates."""
        try:
            self._validation_stats['total_validated'] += 1
            
            # Parse coordinates to standardized format
            lat, lng = self._parse_coordinates(coordinates)
            
            errors = []
            warnings = []
            
            # Basic range validation
            if not self._validate_range(lat, lng, errors):
                self._update_stats('invalid_range')
                return self._create_invalid_result(errors, warnings)
            
            # Precision validation
            if not self._validate_precision(lat, lng, warnings):
                self._update_stats('low_precision')
            
            # H3 validation if enabled
            if self.config.validate_h3:
                if not self._validate_h3(lat, lng, warnings):
                    self._update_stats('invalid_h3')
                    if self.config.strict_mode:
                        errors.append("Failed H3 index generation")
            
            # Land area validation if enabled
            if self.config.check_land_area:
                if not self._validate_land_area(lat, lng, warnings):
                    self._update_stats('water_area')
                    if self.config.strict_mode:
                        errors.append("Coordinates in water area")
            
            # Create normalized result
            normalized = {
                'latitude': round(lat, self.config.require_precision),
                'longitude': round(lng, self.config.require_precision)
            }
            
            if errors:
                self._validation_stats['total_invalid'] += 1
                return self._create_invalid_result(errors, warnings)
            
            self._validation_stats['total_valid'] += 1
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=warnings,
                normalized=normalized
            )
            
        except Exception as e:
            logger.error("Validation error: %s", str(e))
            self._update_stats('validation_error')
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[]
            )
    
    def _parse_coordinates(
        self,
        coordinates: Union[Dict[str, Any], str, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Parse coordinates from various input formats."""
        try:
            if isinstance(coordinates, dict):
                lat = float(coordinates.get('latitude', coordinates.get('lat', 0)))
                lng = float(coordinates.get('longitude', coordinates.get('lng', 0)))
            elif isinstance(coordinates, str):
                coord_dict = json.loads(coordinates.replace("'", '"'))
                lat = float(coord_dict.get('latitude', coord_dict.get('lat', 0)))
                lng = float(coord_dict.get('longitude', coord_dict.get('lng', 0)))
            elif isinstance(coordinates, tuple):
                lat, lng = map(float, coordinates)
            else:
                raise ValidationError(
                    f"Unsupported coordinate format: {type(coordinates)}"
                )
            
            return lat, lng
            
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            raise ValidationError(f"Failed to parse coordinates: {str(e)}")
    
    def _validate_range(
        self,
        lat: float,
        lng: float,
        errors: list[str]
    ) -> bool:
        """Validate coordinate ranges."""
        if not (self.config.lat_min <= lat <= self.config.lat_max):
            errors.append(
                f"Latitude {lat} outside valid range "
                f"[{self.config.lat_min}, {self.config.lat_max}]"
            )
            return False
        
        if not (self.config.lng_min <= lng <= self.config.lng_max):
            errors.append(
                f"Longitude {lng} outside valid range "
                f"[{self.config.lng_min}, {self.config.lng_max}]"
            )
            return False
        
        return True
    
    def _validate_precision(
        self,
        lat: float,
        lng: float,
        warnings: list[str]
    ) -> bool:
        """Validate coordinate precision."""
        lat_precision = len(str(abs(lat)).split('.')[-1])
        lng_precision = len(str(abs(lng)).split('.')[-1])
        
        if lat_precision < self.config.require_precision:
            warnings.append(
                f"Latitude precision ({lat_precision}) below required "
                f"({self.config.require_precision})"
            )
            return False
        
        if lng_precision < self.config.require_precision:
            warnings.append(
                f"Longitude precision ({lng_precision}) below required "
                f"({self.config.require_precision})"
            )
            return False
        
        return True
    
    def _validate_h3(
        self,
        lat: float,
        lng: float,
        warnings: list[str]
    ) -> bool:
        """Validate coordinates using H3 grid system."""
        try:
            h3_index = h3.geo_to_h3(lat, lng, self.config.h3_resolution)
            if not h3_index:
                warnings.append("Failed to generate H3 index")
                return False
            return True
        except Exception as e:
            warnings.append(f"H3 validation error: {str(e)}")
            return False
    
    def _validate_land_area(
        self,
        lat: float,
        lng: float,
        warnings: list[str]
    ) -> bool:
        """Validate if coordinates are on land (placeholder)."""
        # This is a placeholder for land area validation
        # In a real implementation, you would use a proper land/water dataset
        warnings.append("Land area validation not implemented")
        return True
    
    def _create_invalid_result(
        self,
        errors: list[str],
        warnings: list[str]
    ) -> ValidationResult:
        """Create validation result for invalid coordinates."""
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings
        )
    
    def _update_stats(self, error_type: str) -> None:
        """Update validation statistics."""
        self._validation_stats['error_types'][error_type] = (
            self._validation_stats['error_types'].get(error_type, 0) + 1
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validated': self._validation_stats['total_validated'],
            'total_valid': self._validation_stats['total_valid'],
            'total_invalid': self._validation_stats['total_invalid'],
            'error_types': dict(self._validation_stats['error_types']),
            'success_rate': (
                self._validation_stats['total_valid'] /
                self._validation_stats['total_validated']
                if self._validation_stats['total_validated'] > 0 else 0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self._validation_stats = {
            'total_validated': 0,
            'total_valid': 0,
            'total_invalid': 0,
            'error_types': {}
        }

class BatchCoordinateValidator:
    """Handles batch validation of coordinates."""
    
    def __init__(
        self,
        config: Optional[CoordinateValidationConfig] = None,
        batch_size: int = 1000
    ):
        self.validator = CoordinateValidator(config)
        self.batch_size = batch_size
    
    def validate_batch(
        self,
        coordinates: list[Union[Dict[str, Any], str, Tuple[float, float]]]
    ) -> list[ValidationResult]:
        """Validate a batch of coordinates."""
        results = []
        
        for i in range(0, len(coordinates), self.batch_size):
            batch = coordinates[i:i + self.batch_size]
            batch_results = [self.validator.validate(coord) for coord in batch]
            results.extend(batch_results)
            
            # Log batch completion
            valid_count = sum(1 for r in batch_results if r.is_valid)
            logger.info(
                "Batch %d-%d processed: %d/%d valid",
                i,
                i + len(batch),
                valid_count,
                len(batch)
            )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch validation statistics."""
        return self.validator.get_stats()
