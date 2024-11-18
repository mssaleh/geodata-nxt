"""Address validation with support for international formats and standards."""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import pycountry
from langdetect import detect
import phonenumbers

from ..utils.exceptions import ValidationError
from ..core.address import AddressComponents

logger = logging.getLogger(__name__)

@dataclass
class AddressValidationConfig:
    """Configuration for address validation."""
    required_fields: Set[str] = field(default_factory=lambda: {
        'country_code'
    })
    optional_fields: Set[str] = field(default_factory=lambda: {
        'state', 'city', 'postal_code', 'street'
    })
    validate_phone: bool = True
    validate_postal_codes: bool = True
    strict_iso3166: bool = False
    min_address_length: int = 5
    max_address_length: int = 200
    validate_language: bool = False
    allowed_languages: Set[str] = field(default_factory=lambda: {
        'en', 'ar'
    })

@dataclass
class AddressValidationResult:
    """Results of address validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized: Optional[AddressComponents] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AddressValidator:
    """Validates address components and formats."""
    
    def __init__(self, config: Optional[AddressValidationConfig] = None):
        """Initialize validator with configuration."""
        self.config = config or AddressValidationConfig()
        self._initialize_validation_data()
        self._stats = self._initialize_stats()

    def _initialize_validation_data(self) -> None:
        """Initialize validation lookup data."""
        # Country data
        self.countries = {
            country.alpha_2: country 
            for country in pycountry.countries
        }
        
        # Postal code patterns
        self.postal_patterns = {
            'AE': r'^\d{6}$',
            'US': r'^\d{5}(-\d{4})?$',
            'GB': r'^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$',
            'CA': r'^[ABCEGHJKLMNPRSTVXY]\d[ABCEGHJ-NPRSTV-Z] ?\d[ABCEGHJ-NPRSTV-Z]\d$'
        }
        
        # Common abbreviations
        self.street_abbrev = {
            'st': 'street',
            'rd': 'road',
            'ave': 'avenue',
            'blvd': 'boulevard'
        }

    def _initialize_stats(self) -> Dict[str, Any]:
        """Initialize validation statistics."""
        return {
            'total_validated': 0,
            'valid_count': 0,
            'invalid_count': 0,
            'errors_by_type': {},
            'warnings_by_type': {}
        }

    def validate(
        self,
        address: AddressComponents
    ) -> AddressValidationResult:
        """
        Validate address components.
        
        Args:
            address: Address components to validate
            
        Returns:
            Validation result with errors and warnings
        """
        self._stats['total_validated'] += 1
        errors = []
        warnings = []

        try:
            # Required fields validation
            self._validate_required_fields(address, errors)
            
            # Country code validation
            if address.country_code:
                self._validate_country_code(address.country_code, errors)
            
            # State/Province validation
            if address.state and address.country_code:
                self._validate_state(
                    address.state,
                    address.country_code,
                    address.iso3166_2,
                    errors,
                    warnings
                )
            
            # Postal code validation
            if self.config.validate_postal_codes and address.postal_code:
                self._validate_postal_code(
                    address.postal_code,
                    address.country_code,
                    errors,
                    warnings
                )
            
            # Address length validation
            if address.formatted_address:
                self._validate_address_length(
                    address.formatted_address,
                    errors,
                    warnings
                )
            
            # Language validation
            if (self.config.validate_language and
                address.formatted_address):
                self._validate_language(
                    address.formatted_address,
                    errors,
                    warnings
                )

            is_valid = len(errors) == 0
            if is_valid:
                self._stats['valid_count'] += 1
            else:
                self._stats['invalid_count'] += 1

            return AddressValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                normalized=address if is_valid else None,
                metadata=self._generate_metadata(address)
            )

        except Exception as e:
            logger.error("Validation error: %s", str(e))
            return AddressValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=warnings
            )

    def _validate_required_fields(
        self,
        address: AddressComponents,
        errors: List[str]
    ) -> None:
        """Validate presence of required fields."""
        for field in self.config.required_fields:
            if not getattr(address, field):
                self._update_error_stats('missing_required_field')
                errors.append(f"Missing required field: {field}")

    def _validate_country_code(
        self,
        country_code: str,
        errors: List[str]
    ) -> None:
        """Validate country code format and existence."""
        if not re.match(r'^[A-Z]{2}$', country_code):
            self._update_error_stats('invalid_country_code_format')
            errors.append(f"Invalid country code format: {country_code}")
            return

        if country_code not in self.countries:
            self._update_error_stats('unknown_country_code')
            errors.append(f"Unknown country code: {country_code}")

    def _validate_state(
        self,
        state: str,
        country_code: str,
        iso3166_2: Optional[str],
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate state/province information."""
        if self.config.strict_iso3166 and not iso3166_2:
            self._update_error_stats('missing_iso3166_2')
            errors.append("Missing ISO 3166-2 code for state/province")
            return

        if iso3166_2:
            if not re.match(f'^{country_code}-[A-Z0-9]+$', iso3166_2):
                self._update_error_stats('invalid_iso3166_2_format')
                errors.append(f"Invalid ISO 3166-2 format: {iso3166_2}")

            subdivision = pycountry.subdivisions.get(code=iso3166_2)
            if not subdivision:
                self._update_warning_stats('unknown_iso3166_2')
                warnings.append(f"Unknown ISO 3166-2 code: {iso3166_2}")

    def _validate_postal_code(
        self,
        postal_code: str,
        country_code: Optional[str],
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate postal code format."""
        if not country_code:
            self._update_warning_stats('postal_code_no_country')
            warnings.append("Cannot validate postal code without country code")
            return

        pattern = self.postal_patterns.get(country_code)
        if pattern:
            if not re.match(pattern, postal_code, re.I):
                self._update_error_stats('invalid_postal_code')
                errors.append(
                    f"Invalid postal code format for {country_code}: {postal_code}"
                )
        else:
            self._update_warning_stats('unknown_postal_format')
            warnings.append(f"No postal code format defined for {country_code}")

    def _validate_address_length(
        self,
        address: str,
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate address string length."""
        length = len(address)
        if length < self.config.min_address_length:
            self._update_error_stats('address_too_short')
            errors.append(
                f"Address too short ({length} chars, min {self.config.min_address_length})"
            )
        elif length > self.config.max_address_length:
            self._update_error_stats('address_too_long')
            errors.append(
                f"Address too long ({length} chars, max {self.config.max_address_length})"
            )

    def _validate_language(
        self,
        text: str,
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate text language."""
        try:
            detected_lang = detect(text)
            if detected_lang not in self.config.allowed_languages:
                self._update_warning_stats('unsupported_language')
                warnings.append(f"Unsupported language detected: {detected_lang}")
        except Exception as e:
            self._update_warning_stats('language_detection_failed')
            warnings.append(f"Language detection failed: {str(e)}")

    def _generate_metadata(
        self,
        address: AddressComponents
    ) -> Dict[str, Any]:
        """Generate validation metadata."""
        return {
            'validation_level': 'strict' if self.config.strict_iso3166 else 'normal',
            'validated_fields': [
                field for field in self.config.required_fields
                if getattr(address, field)
            ],
            'optional_fields_present': [
                field for field in self.config.optional_fields
                if getattr(address, field)
            ],
            'completeness_score': self._calculate_completeness(address)
        }

    def _calculate_completeness(
        self,
        address: AddressComponents
    ) -> float:
        """Calculate address completeness score."""
        total_fields = len(self.config.required_fields) + len(self.config.optional_fields)
        present_fields = sum(
            1 for field in [*self.config.required_fields, *self.config.optional_fields]
            if getattr(address, field)
        )
        return round(present_fields / total_fields * 100, 2)

    def _update_error_stats(self, error_type: str) -> None:
        """Update error statistics."""
        self._stats['errors_by_type'][error_type] = (
            self._stats['errors_by_type'].get(error_type, 0) + 1
        )

    def _update_warning_stats(self, warning_type: str) -> None:
        """Update warning statistics."""
        self._stats['warnings_by_type'][warning_type] = (
            self._stats['warnings_by_type'].get(warning_type, 0) + 1
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            **self._stats,
            'validation_rate': (
                self._stats['valid_count'] / self._stats['total_validated']
                if self._stats['total_validated'] > 0 else 0
            )
        }

class BatchAddressValidator:
    """Handles batch validation of addresses."""
    
    def __init__(
        self,
        config: Optional[AddressValidationConfig] = None,
        batch_size: int = 1000
    ):
        self.validator = AddressValidator(config)
        self.batch_size = batch_size

    def validate_batch(
        self,
        addresses: List[AddressComponents]
    ) -> List[AddressValidationResult]:
        """Validate a batch of addresses."""
        results = []
        
        for i in range(0, len(addresses), self.batch_size):
            batch = addresses[i:i + self.batch_size]
            batch_results = [self.validator.validate(addr) for addr in batch]
            results.extend(batch_results)
            
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
