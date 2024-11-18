# src/geodata/cli/main.py

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence
import yaml

from ..utils.config import AppConfig
from ..utils.exceptions import ConfigurationError
from ..utils.logging import setup_logging
from ..core.enricher import LocationDataEnricher

logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Geodata enrichment tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    # Required arguments
    parser.add_argument(
        '--input',
        type=str,
        help='Input CSV file path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path'
    )
    
    # Optional enrichment parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Number of records to process in each batch'
    )
    
    parser.add_argument(
        '--h3-resolution',
        type=int,
        choices=range(7, 13),
        help='H3 grid resolution (7-12)'
    )
    
    parser.add_argument(
        '--geometry-format',
        choices=['wkb', 'wkt'],
        help='Output geometry format'
    )
    
    parser.add_argument(
        '--error-handling',
        choices=['strict', 'warn', 'ignore'],
        help='How to handle processing errors'
    )
    
    # Feature flags
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip coordinate validation'
    )
    
    parser.add_argument(
        '--no-address',
        action='store_true',
        help='Skip address enrichment'
    )
    
    parser.add_argument(
        '--no-arabic',
        action='store_true',
        help='Skip Arabic text standardization'
    )
    
    parser.add_argument(
        '--no-uae',
        action='store_true',
        help='Skip UAE-specific parsing'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable response caching'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    
    # Dask options
    parser.add_argument(
        '--scheduler',
        choices=['threads', 'processes', 'single-threaded'],
        help='Dask scheduler type'
    )
    
    parser.add_argument(
        '--n-workers',
        type=int,
        help='Number of Dask workers'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=str,
        help='Memory limit per worker'
    )
    
    return parser

def load_config(args: argparse.Namespace) -> AppConfig:
    """Load and validate configuration from file and/or arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Validated configuration object
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Start with default config if no config file
    config = AppConfig.from_yaml(Path('config/default.yaml'))
    
    # Load config file if specified
    if args.config:
        try:
            config = AppConfig.from_yaml(Path(args.config))
        except ConfigurationError as e:
            logger.error(f"Error loading configuration file: {e}")
            sys.exit(1)
    
    # Convert args to dict, handling feature flags
    args_dict = vars(args)
    if args.no_validate:
        args_dict['validate_coordinates'] = False
    if args.no_address:
        args_dict['enrich_addresses'] = False
    if args.no_arabic:
        args_dict['standardize_arabic'] = False
    if args.no_uae:
        args_dict['add_uae_parsing'] = False
    if args.no_cache:
        args_dict['cache_responses'] = False
    
    # Update config with command line arguments
    config.update_from_args(args_dict)
    
    # Validate final configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise ConfigurationError("Invalid configuration")
    
    return config

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        argv: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    try:
        # Load and validate configuration
        config = load_config(args)
        
        # Setup logging
        setup_logging(config.logging)
        
        # Initialize enricher
        enrich