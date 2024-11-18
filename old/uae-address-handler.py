"""
UAE/Dubai-specific address handling components for geodata.py
---------------------------------------------------------

This module provides specialized handling for UAE addresses with focus on:
- Dubai area/district standardization
- Arabic/English area name mappings
- UAE PO Box validation
- Dubai/UAE address formats
- Common UAE address patterns
"""

import re
from typing import Dict, Optional, Set, List, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class UAEAddressComponents:
    """Specialized components for UAE addresses."""
    emirate: Optional[str] = None
    area: Optional[str] = None
    district: Optional[str] = None
    street_name: Optional[str] = None
    street_number: Optional[str] = None
    building: Optional[str] = None
    floor: Optional[str] = None
    unit: Optional[str] = None
    po_box: Optional[str] = None
    makani: Optional[str] = None
    nearest_landmark: Optional[str] = None

class UAEAddressHandler:
    """
    Specialized handler for UAE addresses with focus on Dubai.
    
    Features:
    - Emirate name standardization
    - Area/district mapping
    - PO Box validation
    - Makani number validation
    - Bilingual support
    """
    
    def __init__(self):
        self._initialize_mappings()
        self._compile_patterns()
    
    def _initialize_mappings(self) -> None:
        """Initialize UAE-specific address mappings."""
        # Emirate names (English-Arabic)
        self.emirate_mappings = {
            'دبي': 'Dubai',
            'ابوظبي': 'Abu Dhabi',
            'أبوظبي': 'Abu Dhabi',
            'الشارقة': 'Sharjah',
            'عجمان': 'Ajman',
            'ام القيوين': 'Umm Al Quwain',
            'أم القيوين': 'Umm Al Quwain',
            'راس الخيمة': 'Ras Al Khaimah',
            'رأس الخيمة': 'Ras Al Khaimah',
            'الفجيرة': 'Fujairah'
        }
        
        # Dubai areas (English-Arabic)
        self.dubai_areas = {
            'البرشاء': 'Al Barsha',
            'القوز': 'Al Quoz',
            'الكرامة': 'Al Karama',
            'المركز التجاري': 'Trade Centre',
            'بر دبي': 'Bur Dubai',
            'ديرة': 'Deira',
            'جميرا': 'Jumeirah',
            'الخليج التجاري': 'Business Bay',
            'مردف': 'Mirdif',
            'القصيص': 'Al Qusais',
            'ند الحمر': 'Nad Al Hamar',
            'جبل علي': 'Jebel Ali',
            'ورسان': 'Warsan',
            'البرشاء جنوب': 'Al Barsha South',
            'ام سقيم': 'Umm Suqeim',
            'أم سقيم': 'Umm Suqeim',
            'الصفوح': 'Al Sufouh',
            'المنارة': 'Al Manara',
            'السطوة': 'Al Satwa',
            'الوصل': 'Al Wasl',
            'هور العنز': 'Hor Al Anz',
            'أبو هيل': 'Abu Hail',
            'المرر': 'Al Murar',
            'نايف': 'Naif',
            'المرقبات': 'Al Muraqqabat',
            'الرقة': 'Al Rigga',
            # Add more mappings as needed
        }
        
        # Common street types
        self.street_types = {
            'شارع': 'Street',
            'طريق': 'Road',
            'سكة': 'Street',
            'ممر': 'Lane',
            'شارع رئيسي': 'Main Street',
            'تقاطع': 'Junction',
            'تفرع': 'Branch'
        }
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for address parsing."""
        # PO Box pattern (e.g., "P.O. Box 123456" or "ص.ب 123456")
        self.po_box_pattern = re.compile(
            r'(?:P\.?O\.?\s*Box|ص\.?\s*ب\.?)\s*(\d+)',
            re.IGNORECASE
        )
        
        # Makani number pattern (8 digits)
        self.makani_pattern = re.compile(r'\b\d{8}\b')
        
        # Street number pattern
        self.street_number_pattern = re.compile(
            r'(?:شارع|street|st\.?)\s*(?:رقم|no\.?|number|#)?\s*(\d+)',
            re.IGNORECASE
        )
        
        # Building pattern
        self.building_pattern = re.compile(
            r'(?:building|bldg\.?|بناية)\s*(?:no\.?|رقم|#)?\s*([^\d]*\d+[^\d]*)',
            re.IGNORECASE
        )
    
    def parse_uae_address(self, address_text: str) -> UAEAddressComponents:
        """
        Parse UAE address text into structured components.
        
        Args:
            address_text: Raw address text (can be in Arabic or English)
            
        Returns:
            Structured UAE address components
        """
        components = UAEAddressComponents()
        
        try:
            # Extract PO Box if present
            po_box_match = self.po_box_pattern.search(address_text)
            if po_box_match:
                components.po_box = po_box_match.group(1)
            
            # Extract Makani number if present
            makani_match = self.makani_pattern.search(address_text)
            if makani_match:
                components.makani = makani_match.group(0)
            
            # Identify emirate
            components.emirate = self._identify_emirate(address_text)
            
            # If Dubai, identify area
            if components.emirate == 'Dubai':
                components.area = self._identify_dubai_area(address_text)
            
            # Extract street information
            street_info = self._extract_street_info(address_text)
            components.street_name = street_info.get('name')
            components.street_number = street_info.get('number')
            
            # Extract building information
            building_info = self._extract_building_info(address_text)
            components.building = building_info.get('name')
            components.floor = building_info.get('floor')
            components.unit = building_info.get('unit')
            
        except Exception as e:
            logger.error(f"Error parsing UAE address: {str(e)}")
        
        return components
    
    def _identify_emirate(self, text: str) -> Optional[str]:
        """Identify emirate from text."""
        # Check Arabic names first
        for arabic, english in self.emirate_mappings.items():
            if arabic in text:
                return english
        
        # Check English names
        for english in self.emirate_mappings.values():
            if english.lower() in text.lower():
                return english
        
        return None
    
    def _identify_dubai_area(self, text: str) -> Optional[str]:
        """Identify Dubai area from text."""
        # Check Arabic names first
        for arabic, english in self.dubai_areas.items():
            if arabic in text:
                return english
        
        # Check English names
        for english in self.dubai_areas.values():
            if english.lower() in text.lower():
                return english
        
        return None
    
    def _extract_street_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract street name and number."""
        result = {'name': None, 'number': None}
        
        # Try to find street number
        street_match = self.street_number_pattern.search(text)
        if street_match:
            result['number'] = street_match.group(1)
        
        # Try to identify street name
        for arabic, english in self.street_types.items():
            if arabic in text or english.lower() in text.lower():
                # Extract the part after the street type
                parts = re.split(f"{arabic}|{english}", text, flags=re.IGNORECASE)
                if len(parts) > 1:
                    # Clean up the street name
                    name = parts[1].strip()
                    name = re.sub(r'^[^\w\u0600-\u06FF]+', '', name)  # Remove leading special chars
                    name = re.sub(r'[^\w\u0600-\u06FF]+$', '', name)  # Remove trailing special chars
                    if name:
                        result['name'] = name
                    break
        
        return result
    
    def _extract_building_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract building details."""
        result = {
            'name': None,
            'floor': None,
            'unit': None
        }
        
        # Try to find building name/number
        building_match = self.building_pattern.search(text)
        if building_match:
            result['name'] = building_match.group(1).strip()
        
        # Try to find floor number
        floor_match = re.search(
            r'(?:floor|الطابق)\s*(?:no\.?|رقم|#)?\s*(\d+)',
            text,
            re.IGNORECASE
        )
        if floor_match:
            result['floor'] = floor_match.group(1)
        
        # Try to find unit number
        unit_match = re.search(
            r'(?:unit|office|flat|apt\.?|شقة|مكتب)\s*(?:no\.?|رقم|#)?\s*(\w+)',
            text,
            re.IGNORECASE
        )
        if unit_match:
            result['unit'] = unit_match.group(1)
        
        return result
    
    def validate_uae_components(self, components: UAEAddressComponents) -> Dict[str, bool]:
        """
        Validate UAE address components.
        
        Args:
            components: UAE address components
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Validate emirate
        results['valid_emirate'] = components.emirate in self.emirate_mappings.values()
        
        # Validate Dubai area if applicable
        results['valid_area'] = (
            components.emirate != 'Dubai' or
            components.area in self.dubai_areas.values()
        )
        
        # Validate PO Box format
        results['valid_po_box'] = (
            not components.po_box or
            (components.po_box.isdigit() and len(components.po_box) <= 6)
        )
        
        # Validate Makani number if present
        results['valid_makani'] = (
            not components.makani or
            (components.makani.isdigit() and len(components.makani) == 8)
        )
        
        return results
    
    def format_uae_address(self, components: UAEAddressComponents) -> str:
        """
        Generate formatted UAE address string.
        
        Args:
            components: UAE address components
            
        Returns:
            Formatted address string
        """
        parts = []
        
        # Add building/unit information
        if components.building:
            building_part = f"Building {components.building}"
            if components.floor:
                building_part += f", Floor {components.floor}"
            if components.unit:
                building_part += f", Unit {components.unit}"
            parts.append(building_part)
        
        # Add street information
        if components.street_name:
            street_part = components.street_name
            if components.street_number:
                street_part = f"Street {components.street_number}, {street_part}"
            parts.append(street_part)
        
        # Add area/district
        if components.area:
            parts.append(components.area)
        if components.district and components.district != components.area:
            parts.append(components.district)
        
        # Add emirate
        if components.emirate:
            parts.append(components.emirate)
        
        # Add country
        parts.append("United Arab Emirates")
        
        # Add PO Box if present
        if components.po_box:
            parts.append(f"P.O. Box {components.po_box}")
        
        # Add Makani if present
        if components.makani:
            parts.append(f"Makani: {components.makani}")
        
        return ", ".join(parts)

