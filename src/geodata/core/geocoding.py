"""Geocoding service integration with rate limiting and caching."""

import time
import random
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from abc import ABC, abstractmethod
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    base_delay: float = 1.0
    jitter: float = 0.5
    burst_size: int = 5
    burst_delay: float = 2.0
    max_retries: int = 3
    retry_delay: float = 1.0

class RateLimiter:
    """Rate limiter with burst handling and jitter."""
    
    def __init__(self, config: RateLimitConfig = RateLimitConfig()):
        self.config = config
        self.last_request_time = 0.0
        self.request_count = 0
        self.total_requests = 0
        self.retry_count = 0
    
    def wait(self) -> None:
        """Wait appropriate time before next request."""
        now = time.time()
        self.request_count += 1
        self.total_requests += 1
        
        # Calculate base wait time
        elapsed = now - self.last_request_time
        base_wait = max(0, self.config.base_delay - elapsed)
        
        # Add jitter
        jitter = random.uniform(0, self.config.jitter)
        
        # Check for burst condition
        if self.request_count >= self.config.burst_size:
            logger.debug(f"Burst detected after {self.request_count} requests")
            wait_time = base_wait + self.config.burst_delay + jitter
            self.request_count = 0
        else:
            wait_time = base_wait + jitter
        
        if wait_time > 0:
            time.sleep(wait_time)
            
        self.last_request_time = time.time()
    
    def reset(self) -> None:
        """Reset rate limiter state."""
        self.last_request_time = 0.0
        self.request_count = 0
        self.retry_count = 0
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get rate limiting statistics."""
        return {
            'total_requests': self.total_requests,
            'current_burst': self.request_count,
            'retry_count': self.retry_count,
            'time_since_last': time.time() - self.last_request_time
        }

class GeocodingError(Exception):
    """Base exception for geocoding errors."""
    pass

class RateLimitError(GeocodingError):
    """Rate limit exceeded error."""
    pass

class ServiceError(GeocodingError):
    """Service request error."""
    pass

class ValidationError(GeocodingError):
    """Response validation error."""
    pass

@dataclass
class GeocodingResponse:
    """Standardized geocoding response."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    provider: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

class GeocodingProvider(ABC):
    """Abstract base class for geocoding providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.rate_limiter = RateLimiter()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @abstractmethod
    def geocode(self, lat: float, lng: float) -> GeocodingResponse:
        """Geocode coordinates to location information."""
        pass
    
    @abstractmethod
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate provider response."""
        pass
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> requests.Response:
        """Make HTTP request with rate limiting and retries."""
        self.rate_limiter.wait()
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                raise RateLimitError(f"Rate limit exceeded: {str(e)}")
            raise ServiceError(f"Request failed: {str(e)}")

class NominatimProvider(GeocodingProvider):
    """OpenStreetMap Nominatim geocoding provider."""
    
    def __init__(self, user_agent: str = "GeoDataEnricher/1.0"):
        super().__init__()
        self.base_url = "https://nominatim.openstreetmap.org/reverse"
        self.session.headers.update({'User-Agent': user_agent})
    
    def geocode(self, lat: float, lng: float) -> GeocodingResponse:
        """Reverse geocode coordinates using Nominatim."""
        params = {
            'format': 'json',
            'lat': lat,
            'lon': lng,
            'zoom': 18,
            'addressdetails': 1
        }
        
        try:
            response = self._make_request(self.base_url, params)
            data = response.json()
            
            if not self.validate_response(data):
                return GeocodingResponse(
                    success=False,
                    error="Invalid response format",
                    provider="nominatim",
                    raw_response=data
                )
            
            standardized = self._standardize_response(data)
            
            return GeocodingResponse(
                success=True,
                data=standardized,
                provider="nominatim",
                raw_response=data
            )
            
        except (RateLimitError, ServiceError) as e:
            return GeocodingResponse(
                success=False,
                error=str(e),
                provider="nominatim"
            )
        
        except Exception as e:
            logger.error(f"Unexpected error in Nominatim geocoding: {str(e)}")
            return GeocodingResponse(
                success=False,
                error=f"Unexpected error: {str(e)}",
                provider="nominatim"
            )
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate Nominatim response format."""
        required_fields = {'lat', 'lon', 'address'}
        if not all(field in response for field in required_fields):
            return False
            
        if not isinstance(response.get('address'), dict):
            return False
            
        return True
    
    def _standardize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize Nominatim response format."""
        address = response.get('address', {})
        
        return {
            'country': address.get('country'),
            'country_code': address.get('country_code', '').upper(),
            'state': address.get('state') or address.get('region'),
            'city': address.get('city') or address.get('town') or address.get('village'),
            'district': address.get('suburb') or address.get('district'),
            'postal_code': address.get('postcode'),
            'formatted_address': response.get('display_name'),
            'location_type': response.get('type'),
            'confidence': response.get('importance', 0) * 100
        }

class GeocodingService:
    """Main service for handling geocoding requests with caching."""
    
    def __init__(self, providers: Optional[List[GeocodingProvider]] = None):
        self.providers = providers or [NominatimProvider()]
        self._cache: Dict[str, GeocodingResponse] = {}
    
    def geocode(self, lat: float, lng: float, use_cache: bool = True) -> GeocodingResponse:
        """Geocode coordinates using available providers."""
        cache_key = f"{lat:.6f},{lng:.6f}"
        
        if use_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for coordinates: {cache_key}")
            return self._cache[cache_key]
        
        for provider in self.providers:
            try:
                response = provider.geocode(lat, lng)
                if response.success:
                    if use_cache:
                        self._cache[cache_key] = response
                    return response
                    
                logger.warning(
                    f"Provider {provider.__class__.__name__} failed: {response.error}"
                )
                
            except Exception as e:
                logger.error(
                    f"Error with provider {provider.__class__.__name__}: {str(e)}"
                )
                continue
        
        return GeocodingResponse(
            success=False,
            error="All providers failed",
            provider="all"
        )
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cache),
            'memory_usage': sum(len(str(v)) for v in self._cache.values())
        }
