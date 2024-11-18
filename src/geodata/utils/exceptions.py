"""Custom exceptions for the geodata enrichment toolkit."""

class GeodataError(Exception):
    """Base exception for all geodata-related errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class ConfigurationError(GeodataError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_path: str = None):
        details = {'config_path': config_path} if config_path else {}
        super().__init__(f"Configuration error: {message}", details)

class ValidationError(GeodataError):
    """Data validation errors."""
    
    def __init__(self, message: str, field: str = None, value: str = None):
        details = {
            'field': field,
            'value': value
        } if field else {}
        super().__init__(f"Validation error: {message}", details)

class EnrichmentError(GeodataError):
    """Data enrichment process errors."""
    
    def __init__(self, message: str, record_id: str = None):
        details = {'record_id': record_id} if record_id else {}
        super().__init__(f"Enrichment error: {message}", details)

class CoordinateError(ValidationError):
    """Coordinate-specific validation errors."""
    
    def __init__(
        self,
        message: str,
        latitude: float = None,
        longitude: float = None
    ):
        details = {
            'latitude': latitude,
            'longitude': longitude
        } if latitude is not None and longitude is not None else {}
        super().__init__(f"Coordinate error: {message}", details=details)

class AddressError(ValidationError):
    """Address-specific validation errors."""
    
    def __init__(
        self,
        message: str,
        component: str = None,
        value: str = None,
        country_code: str = None
    ):
        details = {
            'component': component,
            'value': value,
            'country_code': country_code
        }
        super().__init__(f"Address error: {message}", details=details)

class GeocodingError(GeodataError):
    """Geocoding service errors."""
    
    def __init__(self, message: str, provider: str = None, status_code: int = None):
        details = {
            'provider': provider,
            'status_code': status_code
        }
        super().__init__(f"Geocoding error: {message}", details=details)

class RateLimitError(GeocodingError):
    """Rate limit exceeded errors."""
    
    def __init__(self, message: str, provider: str, retry_after: int = None):
        details = {
            'provider': provider,
            'retry_after': retry_after
        }
        super().__init__(f"Rate limit exceeded: {message}", details=details)

class DataProcessingError(GeodataError):
    """Data processing and transformation errors."""
    
    def __init__(
        self,
        message: str,
        stage: str = None,
        input_data: str = None
    ):
        details = {
            'processing_stage': stage,
            'input_data': input_data
        }
        super().__init__(f"Processing error: {message}", details=details)

class ResourceError(GeodataError):
    """Resource access and availability errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: str = None,
        resource_path: str = None
    ):
        details = {
            'resource_type': resource_type,
            'resource_path': resource_path
        }
        super().__init__(f"Resource error: {message}", details=details)

class CacheError(GeodataError):
    """Cache operation errors."""
    
    def __init__(
        self,
        message: str,
        operation: str = None,
        cache_key: str = None
    ):
        details = {
            'operation': operation,
            'cache_key': cache_key
        }
        super().__init__(f"Cache error: {message}", details=details)

class DaskError(GeodataError):
    """Dask-specific processing errors."""
    
    def __init__(
        self,
        message: str,
        scheduler: str = None,
        task_name: str = None
    ):
        details = {
            'scheduler': scheduler,
            'task_name': task_name
        }
        super().__init__(f"Dask error: {message}", details=details)

class UAEAddressError(AddressError):
    """UAE-specific address handling errors."""
    
    def __init__(
        self,
        message: str,
        component: str = None,
        value: str = None,
        emirate: str = None
    ):
        details = {
            'component': component,
            'value': value,
            'emirate': emirate
        }
        super().__init__(f"UAE address error: {message}", details=details)

def format_error_details(error: GeodataError) -> str:
    """Format error details for logging or display."""
    parts = [error.message]
    
    if error.details:
        details_str = "; ".join(
            f"{k}={v}" for k, v in error.details.items() if v is not None
        )
        if details_str:
            parts.append(f"Details: {details_str}")
    
    return " | ".join(parts)

def handle_processing_error(
    error: Exception,
    logger,
    strict_mode: bool = False,
    record_id: str = None
) -> None:
    """Unified error handling for processing errors."""
    if isinstance(error, GeodataError):
        error_message = format_error_details(error)
    else:
        error_message = str(error)
    
    if record_id:
        error_message = f"Record {record_id}: {error_message}"
    
    if strict_mode:
        logger.error(error_message)
        raise error
    else:
        logger.warning(error_message)
