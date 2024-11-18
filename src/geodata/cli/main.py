"""Command-line interface for geodata enrichment toolkit."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ..core.enricher import create_enricher
from ..utils.config import AppConfig
from ..utils.exceptions import ConfigurationError, GeodataError
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)
console = Console()

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Geodata enrichment toolkit",
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
    
    # Processing options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of records to process in each batch'
    )
    
    parser.add_argument(
        '--h3-resolution',
        type=int,
        choices=range(7, 13),
        default=9,
        help='H3 grid resolution (7-12)'
    )
    
    parser.add_argument(
        '--geometry-format',
        choices=['wkb', 'wkt', 'both'],
        default='wkb',
        help='Output geometry format'
    )
    
    parser.add_argument(
        '--error-handling',
        choices=['strict', 'warn', 'ignore'],
        default='warn',
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
    
    # Dask options
    parser.add_argument(
        '--scheduler',
        choices=['threads', 'processes', 'single-threaded'],
        default='threads',
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
        help='Memory limit per worker (e.g., "2GB")'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )
    
    return parser

def load_config(args: argparse.Namespace) -> AppConfig:
    """Load and validate configuration."""
    try:
        # Start with default config
        if args.config:
            config = AppConfig.from_yaml(Path(args.config))
        else:
            # Load default config
            default_config_path = Path(__file__).parent.parent / 'config' / 'default.yaml'
            if default_config_path.exists():
                config = AppConfig.from_yaml(default_config_path)
            else:
                raise ConfigurationError("Default configuration not found")
        
        # Update with command line arguments
        args_dict = vars(args)
        
        # Handle feature flags
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
        
        config.update_from_args(args_dict)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            for error in errors:
                console.print(f"[red]Configuration error: {error}")
            raise ConfigurationError("Invalid configuration")
        
        return config
        
    except Exception as e:
        console.print(f"[red]Error loading configuration: {str(e)}")
        sys.exit(1)

def process_data(config: AppConfig, show_progress: bool = True) -> None:
    """Process data using configured enricher."""
    try:
        enricher = create_enricher(config.enrichment)
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
            disable=not show_progress
        ) as progress:
            task = progress.add_task("Processing data...", total=None)
            enricher.process_file()
            progress.update(task, completed=True)
        
        # Log final statistics
        stats = enricher.get_stats()
        console.print("\n[green]Processing completed successfully!")
        console.print(f"Total records processed: {stats['total_records']}")
        console.print(f"Successfully processed: {stats['successful_records']}")
        console.print(f"Failed records: {stats['failed_records']}")
        console.print(
            f"Success rate: {stats['success_rate']:.2f}%"
        )
        
    except Exception as e:
        console.print(f"[red]Error during processing: {str(e)}")
        if config.logging.level == 'DEBUG':
            console.print_exception()
        sys.exit(1)

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    try:
        # Load and validate configuration
        config = load_config(args)
        
        # Setup logging
        setup_logging(config.logging)
        
        # Log startup information
        logger.info(
            "Starting geodata enrichment:\n"
            "Input file: %s\n"
            "Output file: %s\n"
            "Batch size: %d\n"
            "H3 resolution: %d\n"
            "Error handling: %s",
            config.enrichment.input_file,
            config.enrichment.output_file,
            config.enrichment.batch_size,
            config.enrichment.h3_resolution,
            config.enrichment.error_handling
        )
        
        # Process data
        process_data(config, show_progress=not args.no_progress)
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user")
        return 130
        
    except GeodataError as e:
        console.print(f"[red]Error: {str(e)}")
        return 1
        
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}")
        if args.log_level == 'DEBUG':
            console.print_exception()
        return 1

if __name__ == "__main__":
    sys.exit(main())
