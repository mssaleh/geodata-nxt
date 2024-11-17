"""
Address standardization components for geodata.py
----------------------------------------------

This module provides functionality for standardizing address data,
with special handling for Arabic text, mixed-language content,
and inconsistent address formats.

Features:
- Arabic text detection and transliteration
- Address component normalization
- Mixed language handling
- ISO-3166 code standardization
- Format validation
"""

import re
import json
from typing import Dict, Optional, List, Union, Set
from dataclasses import dataclass
import pycountry
from langdetect import detect, DetectorFactory
import arabic_reshaper
from bidi.algorithm import get_display
import logging
from collections import defaultdict
from fuzzywuzzy import fuzz
from unidecode import unidecode

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AddressComponents:
    """Standardized address components."""
    country: Optional[str] = None
    country_code: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    street: Optional[str] = None
    building: Optional[str] = None
    postal_code: Optional[str] = None
    formatted_address: Optional[str] = None
    iso3166_2: Optional[str] = None

class AddressStandardizer:
    """
    Handles standardization of address data with special focus on
    Arabic text and mixed language content.
    """
    
    def __init__(self):
        self._load_country_data()
        self._initialize_cached_data()
    
    def _load_country_data(self) -> None:
        """Initialize country and subdivision data from pycountry."""
        self.countries = {country.alpha_2: country for country in pycountry.countries}
        self.subdivisions = defaultdict(dict)
        
        for subdivision in pycountry.subdivisions:
            self.subdivisions[subdivision.country_code][subdivision.code] = subdivision
            
        # Add common Arabic country names
        self.arabic_country_mappings = {
            'الإمارات': 'AE',
            'الامارات': 'AE',
            'السعودية': 'SA',
            'مصر': 'EG',
            'قطر': 'QA',
            'البحرين': 'BH',
            'عمان': 'OM',
            'الكويت': 'KW',
            # Add more mappings as needed
        }
    
    def _initialize_cached_data(self) -> None:
        """Initialize cached data for faster processing."""
        self.cached_translations: Dict[str, str] = {}
        self.cached_iso_codes: Dict[str, str] = {}
    
    def standardize_address(self, address_data: Dict[str, Any]) -> AddressComponents:
        """
        Standardize address components from raw data.
        
        Args:
            address_data: Raw address data dictionary
            
        Returns:
            Standardized AddressComponents object
        """
        # Create base components
        components = AddressComponents()
        
        try:
            # Handle country and country code
            country_info = self._standardize_country(
                address_data.get('country'),
                address_data.get('country_code')
            )
            components.country = country_info['country']
            components.country_code = country_info['country_code']
            
            # Handle state/province
            state_info = self._standardize_state(
                address_data.get('state'),
                components.country_code
            )
            components.state = state_info['state']
            components.iso3166_2 = state_info['iso3166_2']
            
            # Handle city
            components.city = self._standardize_city(
                address_data.get('city'),
                components.country_code,
                components.state
            )
            
            # Handle district
            components.district = self._standardize_district(
                address_data.get('ward') or address_data.get('district'),
                components.city
            )
            
            # Handle street
            components.street = self._standardize_street(address_data.get('street'))
            
            # Handle postal code
            components.postal_code = self._standardize_postal_code(
                address_data.get('postal_code'),
                components.country_code
            )
            
            # Generate formatted address
            components.formatted_address = self._generate_formatted_address(components)
            
        except Exception as e:
            logger.error(f"Error standardizing address: {str(e)}")
            
        return components
    
    def _standardize_country(
        self,
        country: Optional[str],
        country_code: Optional[str]
    ) -> Dict[str, Optional[str]]:
        """
        Standardize country information.
        
        Args:
            country: Country name (can be in Arabic or English)
            country_code: ISO country code
            
        Returns:
            Dictionary with standardized country name and code
        """
        result = {'country': None, 'country_code': None}
        
        if not country and not country_code:
            return result
            
        try:
            # If we have a valid country code, use it
            if country_code:
                country_code = country_code.upper()
                if country_code in self.countries:
                    result['country_code'] = country_code
                    result['country'] = self.countries[country_code].name
                    return result
            
            # If we have a country name, try to match it
            if country:
                # Clean and standardize the text
                cleaned_country = self._clean_text(country)
                
                # Check Arabic mappings
                if cleaned_country in self.arabic_country_mappings:
                    code = self.arabic_country_mappings[cleaned_country]
                    result['country_code'] = code
                    result['country'] = self.countries[code].name
                    return result
                
                # Try fuzzy matching with English names
                best_match = None
                best_score = 0
                
                for c in pycountry.countries:
                    score = fuzz.ratio(cleaned_country.lower(), c.name.lower())
                    if score > best_score and score > 80:  # Threshold for matching
                        best_score = score
                        best_match = c
                
                if best_match:
                    result['country_code'] = best_match.alpha_2
                    result['country'] = best_match.name
            
        except Exception as e:
            logger.warning(f"Error standardizing country: {str(e)}")
            
        return result
    
    def _standardize_state(
        self,
        state: Optional[str],
        country_code: Optional[str]
    ) -> Dict[str, Optional[str]]:
        """
        Standardize state/province information.
        
        Args:
            state: State/province name
            country_code: ISO country code
            
        Returns:
            Dictionary with standardized state name and ISO 3166-2 code
        """
        result = {'state': None, 'iso3166_2': None}
        
        if not state or not country_code:
            return result
            
        try:
            # Clean the state name
            cleaned_state = self._clean_text(state)
            
            # Check cache first
            cache_key = f"{country_code}:{cleaned_state}"
            if cache_key in self.cached_iso_codes:
                return self.cached_iso_codes[cache_key]
            
            # Get subdivisions for the country
            country_subdivisions = self.subdivisions[country_code]
            
            # Try to find the best match
            best_match = None
            best_score = 0
            
            for code, subdivision in country_subdivisions.items():
                # Try both name and local name if available
                names_to_try = [subdivision.name]
                if hasattr(subdivision, 'local_name'):
                    names_to_try.append(subdivision.local_name)
                
                for name in names_to_try:
                    score = fuzz.ratio(cleaned_state.lower(), name.lower())
                    if score > best_score and score > 80:  # Threshold for matching
                        best_score = score
                        best_match = subdivision
            
            if best_match:
                result['state'] = best_match.name
                result['iso3166_2'] = best_match.code
                
                # Cache the result
                self.cached_iso_codes[cache_key] = result
                
        except Exception as e:
            logger.warning(f"Error standardizing state: {str(e)}")
            
        return result
    
    def _standardize_city(
        self,
        city: Optional[str],
        country_code: Optional[str],
        state: Optional[str]
    ) -> Optional[str]:
        """
        Standardize city name.
        
        Args:
            city: City name
            country_code: ISO country code
            state: State/province name
            
        Returns:
            Standardized city name
        """
        if not city:
            return None
            
        try:
            # Clean and standardize the text
            cleaned_city = self._clean_text(city)
            
            # Additional standardization logic can be added here
            # For example, checking against a database of city names
            
            return cleaned_city
            
        except Exception as e:
            logger.warning(f"Error standardizing city: {str(e)}")
            return city
    
    def _standardize_district(
        self,
        district: Optional[str],
        city: Optional[str]
    ) -> Optional[str]:
        """Standardize district/ward name."""
        if not district:
            return None
            
        try:
            return self._clean_text(district)
        except Exception as e:
            logger.warning(f"Error standardizing district: {str(e)}")
            return district
    
    def _standardize_street(self, street: Optional[str]) -> Optional[str]:
        """Standardize street address."""
        if not street:
            return None
            
        try:
            return self._clean_text(street)
        except Exception as e:
            logger.warning(f"Error standardizing street: {str(e)}")
            return street
    
    def _standardize_postal_code(
        self,
        postal_code: Optional[str],
        country_code: Optional[str]
    ) -> Optional[str]:
        """Standardize postal code format."""
        if not postal_code:
            return None
            
        try:
            # Remove any whitespace and non-alphanumeric characters
            cleaned = re.sub(r'[^A-Za-z0-9]', '', str(postal_code))
            
            # Add country-specific postal code formatting here if needed
            # For example, UAE postal code format
            if country_code == 'AE':
                if len(cleaned) != 6:
                    return None
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error standardizing postal code: {str(e)}")
            return postal_code
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and standardize text, handling both Arabic and Latin scripts.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned and standardized text
        """
        if not text:
            return ""
            
        try:
            # Detect text language
            lang = detect(text)
            
            if lang == 'ar':
                # Handle Arabic text
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                return bidi_text
            else:
                # Handle Latin script
                # Remove extra whitespace
                cleaned = ' '.join(text.split())
                # Normalize Unicode characters
                cleaned = unidecode(cleaned)
                return cleaned
                
        except Exception as e:
            logger.warning(f"Error cleaning text: {str(e)}")
            return text
    
    def _generate_formatted_address(self, components: AddressComponents) -> str:
        """
        Generate a formatted address string from components.
        
        Args:
            components: AddressComponents object
            
        Returns:
            Formatted address string
        """
        parts = []
        
        # Add building if available
        if components.building:
            parts.append(components.building)
        
        # Add street
        if components.street:
            parts.append(components.street)
        
        # Add district
        if components.district:
            parts.append(components.district)
        
        # Add city
        if components.city:
            parts.append(components.city)
        
        # Add state
        if components.state:
            parts.append(components.state)
        
        # Add postal code
        if components.postal_code:
            parts.append(components.postal_code)
        
        # Add country
        if components.country:
            parts.append(components.country)
        
        return ', '.join(filter(None, parts))

class AddressValidator:
    """
    Validates standardized address components.
    """
    
    def validate_components(self, components: AddressComponents) -> Dict[str, bool]:
        """
        Validate each address component.
        
        Args:
            components: AddressComponents object
            
        Returns:
            Dictionary of validation results for each component
        """
        results = {}
        
        # Validate country code
        results['country_code'] = (
            bool(components.country_code) and
            len(components.country_code) == 2 and
            components.country_code.isalpha()
        )
        
        # Validate ISO 3166-2 code
        results['iso3166_2'] = bool(
            components.iso3166_2 and
            re.match(r'^[A-Z]{2}-[A-Z0-9]+$', components.iso3166_2)
        )
        
        # Validate presence of essential components
        results['has_city'] = bool(components.city)
        results['has_state'] = bool(components.state)
        
        # Validate consistency
        results['country_state_match'] = (
            bool(components.country_code) and
            bool(components.iso3166_2) and
            components.iso3166_2.startswith(components.country_code)
        )
        
        return results
