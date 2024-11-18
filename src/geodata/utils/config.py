"""Configuration management for geodata enrichment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
import yaml
import logging
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

@dataclass
class EnrichmentConfig:
    """Core enrichment process configuration."""
    input_file: Path
    output_file: Path
    batch_size: int = 100
    h3_resolution: int = 9
    progress_log_interval: int = 1000
    validate_coordinates: bool = True
    enrich_addresses: bool = True
    standardize_arabic: bool = True
    add_uae_parsing: bool = True
    cache_responses: bool = True
    geometry_format: str = "wkb"
    error_handling: str = "warn"

@dataclass
class ValidationConfig:
    """Validation rules configuration."""
    coordinates: Dict[str, Any] = field(default_factory=lambda: {
        "lat_min": -90.0,
        "lat_max": 90.0,
        "lng_min": -180.0,
        "lng_max": 180.0,
        "require_precision": 6
    })
    
    address: Dict[str, Any] = field(default_factory=lambda: {
        "required_fields": ["country_code"],
        "postal_code_validation": True,
        "normalize_abbreviations": True,
        "strict_iso3166": False
    })

@dataclass
class GeocodingConfig:
    """Geocoding service configuration."""
    nominatim: Dict[str, Any] = field(default_factory=lambda: {
        "user_agent": "GeoDataEnricher/1.0",
        "base_url": "https://nominatim.openstreetmap.org/reverse",
        "timeout": 10,
        "max_retries": 3,
        "retry_delay": 1.0,
        "zoom": 18,
        "address_details": True
    })
    
    rate_limiting: Dict[str, Any] = field(default_factory=lambda: {
        "base_delay": 1.0,
        "jitter": 0.5,
        "burst_size": 5,
        "burst_delay": 2.0,
        "max_retries": 3,
        "retry_delay": 1.0
    })
    
    http: Dict[str, Any] = field(default_factory=lambda: {
        "retry_strategy": {
            "total": 3,
            "backoff_factor": 0.5,
            "status_forcelist": [429, 500, 502, 503, 504],
            "allowed_methods": ["GET"]
        }
    })

@dataclass
class UAEAddressConfig:
    """UAE-specific address handling configuration."""
    emirate_matching_threshold: int = 80
    area_matching_threshold: int = 80
    po_box_validation: bool = True
    makani_validation: bool = True
    standardize_area_names: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file: Optional[str] = None
    console: bool = True
    rich_formatting: bool = True
    rich_tracebacks: bool = True
    log_file_timestamp: bool = True
    rotation: Optional[Dict[str, Any]] = None

@dataclass
class DaskConfig:
    """Dask distributed computing configuration."""
    cluster: Dict[str, Any] = field(default_factory=lambda: {
        "type": "local",
        "scheduler": "threads",
        "n_workers": None,
        "threads_per_worker": 1,
        "memory_limit": "0",
        "processes": True,
        "dashboard_address": ":8787",
        "local_directory": None
    })
    
    compute: Dict[str, Any] = field(default_factory=lambda: {
        "blocksize": "64MB",
        "partition_size": "64MB",
        "assume_missing": True,
        "dtype_backend": "numpy",
        "low_memory": False
    })
    
    advanced: Dict[str, Any] = field(default_factory=lambda: {
        "retries": 3,
        "retry_delay": 1,
        "worker_startup_timeout": 60,
        "worker_death_timeout": 60
    })

@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    type: str = "memory"
    max_size: int = 10000
    ttl: int = 3600
    directory: str = ".cache"
    compress: bool = False
    protocol: str = "pickle"

@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""
    chunk_size: int = 10000
    min_partition_size: str = "32MB"
    max_partition_size: str = "128MB"
    compression: Optional[str] = None
    write_mode: str = "single_file"
    optimize_dtypes: bool = True
    use_thread_pool: bool = True
    thread_pool_size: Optional[int] = None
    
    progress: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "update_interval": 1000,
        "spinner": True,
        "time_elapsed": True,
        "rich_progress": True
    })

@dataclass
class AppConfig:
    """Main application configuration."""
    enrichment: EnrichmentConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    geocoding: GeocodingConfig = field(default_factory=GeocodingConfig)
    uae_address: UAEAddressConfig = field(default_factory=UAEAddressConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dask: DaskConfig = field(default_factory=DaskConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    development: Dict[str, Any] = field(default_factory=lambda: {
        "debug": False,
        "verbose": False,
        "profile": False,
        "profile_output": "profile.stats",
        "raise_warnings": False
    })

    @classmethod
    def from_yaml(cls, path: Path) -> 'AppConfig':
        """Create configuration from YAML file."""
        try:
            with open(path) as f:
                config_dict = yaml.safe_load(f)
            
            # Convert paths to Path objects
            if 'enrichment' in config_dict:
                for key in ['input_file', 'output_file']:
                    if key in config_dict['enrichment']:
                        config_dict['enrichment'][key] = Path(
                            config_dict['enrichment'][key]
                        )
            
            return cls(
                enrichment=EnrichmentConfig(**config_dict.get('enrichment', {})),
                validation=ValidationConfig(**config_dict.get('validation', {})),
                geocoding=GeocodingConfig(**config_dict.get('geocoding', {})),
                uae_address=UAEAddressConfig(**config_dict.get('uae_address', {})),
                logging=LoggingConfig(**config_dict.get('logging', {})),
                dask=DaskConfig(**config_dict.get('dask', {})),
                cache=CacheConfig(**config_dict.get('cache', {})),
                performance=PerformanceConfig(**config_dict.get('performance', {})),
                development=config_dict.get('development', {})
            )
            
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")

    def update_from_args(self, args: Dict[str, Any]) -> None:
        """Update configuration from command line arguments."""
        # Update enrichment config
        enrichment_args = {
            k: v for k, v in args.items() 
            if hasattr(self.enrichment, k) and v is not None
        }
        for k, v in enrichment_args.items():
            setattr(self.enrichment, k, v)
        
        # Update logging config
        if args.get('log_level'):
            self.logging.level = args['log_level']
        if args.get('log_file'):
            self.logging.file = args['log_file']
        
        # Update Dask config
        if args.get('scheduler'):
            self.dask.cluster['scheduler'] = args['scheduler']
        if args.get('n_workers'):
            self.dask.cluster['n_workers'] = args['n_workers']
        if args.get('memory_limit'):
            self.dask.cluster['memory_limit'] = args['memory_limit']

    def validate(self) -> List[str]:
        """Validate configuration settings."""
        errors = []
        
        # Validate enrichment config
        if not isinstance(self.enrichment.input_file, Path):
            errors.append("Input file must be a Path object")
        elif not self.enrichment.input_file.exists():
            errors.append(f"Input file not found: {self.enrichment.input_file}")
            
        if not 7 <= self.enrichment.h3_resolution <= 12:
            errors.append("H3 resolution must be between 7 and 12")
            
        if self.enrichment.geometry_format not in ['wkb', 'wkt', 'both']:
            errors.append("Geometry format must be 'wkb', 'wkt', or 'both'")
            
        if self.enrichment.error_handling not in ['strict', 'warn', 'ignore']:
            errors.append("Error handling must be 'strict', 'warn', or 'ignore'")
        
        # Validate logging config
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.logging.level not in valid_levels:
            errors.append(f"Invalid logging level: {self.logging.level}")
        
        # Validate Dask config
        valid_schedulers = {'threads', 'processes', 'single-threaded'}
        if self.dask.cluster['scheduler'] not in valid_schedulers:
            errors.append(f"Invalid scheduler type: {self.dask.cluster['scheduler']}")
        
        if self.dask.cluster['n_workers'] is not None:
            if not isinstance(self.dask.cluster['n_workers'], int):
                errors.append("Number of workers must be an integer")
            elif self.dask.cluster['n_workers'] < 1:
                errors.append("Number of workers must be positive")
        
        return errors

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"AppConfig("
            f"enrichment={self.enrichment}, "
            f"validation=ValidationConfig(...), "
            f"geocoding=GeocodingConfig(...), "
            f"logging={self.logging.level})"
        )
