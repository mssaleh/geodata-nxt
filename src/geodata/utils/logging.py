"""Logging configuration and utilities for geodata enrichment."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install as install_rich_traceback
from dataclasses import asdict

from .exceptions import ConfigurationError
from ..utils.config import LoggingConfig

# Initialize rich console
console = Console()

class GeodataJsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(
        self,
        additional_fields: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.additional_fields = additional_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }

        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        # Add configured additional fields
        log_data.update(self.additional_fields)

        return json.dumps(log_data)

class GeodataLogger:
    """Custom logger with enhanced functionality."""
    
    def __init__(
        self,
        name: str,
        config: LoggingConfig
    ):
        self.logger = logging.getLogger(name)
        self.config = config
        self._configure_logger()

    def _configure_logger(self) -> None:
        """Configure logger based on settings."""
        self.logger.setLevel(self.config.level)
        self._add_handlers()

    def _add_handlers(self) -> None:
        """Add configured handlers to logger."""
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Console handler with rich formatting if enabled
        if self.config.console:
            console_handler = (
                RichHandler(
                    console=console,
                    show_time=True,
                    show_path=True,
                    enable_link_path=True,
                    rich_tracebacks=self.config.rich_tracebacks,
                    tracebacks_show_locals=True
                )
                if self.config.rich_formatting
                else logging.StreamHandler(sys.stdout)
            )
            
            console_handler.setFormatter(
                logging.Formatter(self.config.format, self.config.date_format)
            )
            self.logger.addHandler(console_handler)

        # File handler if configured
        if self.config.file:
            file_path = self._get_log_file_path()
            try:
                file_handler = logging.FileHandler(file_path)
                file_handler.setFormatter(GeodataJsonFormatter())
                self.logger.addHandler(file_handler)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to create log file handler: {str(e)}"
                )

    def _get_log_file_path(self) -> Path:
        """Generate log file path with optional timestamp."""
        log_path = Path(self.config.file)
        
        if self.config.log_file_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return log_path.with_name(
                f"{log_path.stem}_{timestamp}{log_path.suffix}"
            )
        
        return log_path

    def log_validation_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        """Log validation error with context."""
        self.logger.error(
            "Validation error",
            extra={'extra_fields': {
                'error_type': error.__class__.__name__,
                'error_message': str(error),
                'validation_context': context
            }}
        )

    def log_processing_stats(
        self,
        stats: Dict[str, Any]
    ) -> None:
        """Log processing statistics."""
        self.logger.info(
            "Processing statistics",
            extra={'extra_fields': {'statistics': stats}}
        )

    def log_geocoding_response(
        self,
        response: Any,
        context: Dict[str, Any]
    ) -> None:
        """Log geocoding service response."""
        self.logger.debug(
            "Geocoding response received",
            extra={'extra_fields': {
                'provider': getattr(response, 'provider', 'unknown'),
                'success': getattr(response, 'success', False),
                'context': context
            }}
        )

def setup_logging(config: Union[LoggingConfig, Dict[str, Any]]) -> None:
    """Setup logging configuration."""
    if not isinstance(config, LoggingConfig):
        config = LoggingConfig(**config)

    # Configure rich tracebacks if enabled
    if config.rich_tracebacks:
        install_rich_traceback(
            show_locals=True,
            suppress=[
                'click',
                'rich',
                'pandas',
                'numpy',
                'dask'
            ]
        )

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger level
    root_logger.setLevel(config.level)

    # Configure console logging
    if config.console:
        console_handler = (
            RichHandler(
                console=console,
                show_time=True,
                show_path=True,
                enable_link_path=True,
                rich_tracebacks=config.rich_tracebacks,
                tracebacks_show_locals=True
            )
            if config.rich_formatting
            else logging.StreamHandler(sys.stdout)
        )
        
        console_handler.setFormatter(
            logging.Formatter(config.format, config.date_format)
        )
        root_logger.addHandler(console_handler)

    # Configure file logging if specified
    if config.file:
        try:
            log_path = Path(config.file)
            if config.log_file_timestamp:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_path = log_path.with_name(
                    f"{log_path.stem}_{timestamp}{log_path.suffix}"
                )
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(GeodataJsonFormatter())
            root_logger.addHandler(file_handler)
            
            logging.info("File logging initialized: %s", log_path)
            
        except Exception as e:
            console.print(
                f"[red]Warning: Failed to initialize file logging: {str(e)}"
            )

    logging.info("Logging system initialized")

class LoggerFactory:
    """Factory for creating configured loggers."""
    
    @staticmethod
    def get_logger(
        name: str,
        config: Optional[LoggingConfig] = None
    ) -> GeodataLogger:
        """Get a configured logger instance."""
        if config is None:
            config = LoggingConfig()
        return GeodataLogger(name, config)

def get_logger(
    name: str,
    config: Optional[LoggingConfig] = None
) -> GeodataLogger:
    """Convenience function to get a configured logger."""
    return LoggerFactory.get_logger(name, config)
