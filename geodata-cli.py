"""
Command-line interface for geodata.py
-----------------------------------

This module provides a comprehensive CLI for the geodata enrichment pipeline.
It handles argument parsing, configuration management, and pipeline execution.

Usage:
    python geodata.py enrich --input places.csv --output enriched_places.csv [options]
    python geodata.py validate --input places.csv [options]
    python geodata.py stats --input places.csv [options]
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.logging import RichHandler
from dataclasses import dataclass, asdict

from .enrichment import LocationDataEnricher, EnrichmentConfig, ProcessingStats

# Initialize rich console for better output
console = Console()

def setup_logging(log_level: str, log_file: Optional[str] = None) -> None:
    """Configure logging with rich output."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True),
            *([] if not log_file else [
                logging.FileHandler(
                    f"{log_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            ])
        ]
    )

class GeodataCLI:
    """
    Command-line interface for geodata enrichment pipeline.
    
    Features:
    - Multiple commands (enrich, validate, stats)
    - Rich progress display
    - Configuration file support
    - Detailed logging options
    """
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser."""
        parser = argparse.ArgumentParser(
            description="Geodata enrichment pipeline CLI",
            formatter_class=argparse.RichHelpFormatter
        )
        
        # Add global options
        parser.add_argument(
            '--config',
            type=str,
            help='Path to YAML configuration file'
        )
        
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO',
            help='Set logging level'
        )
        
        parser.add_argument(
            '--log-file',
            type=str,
            help='Base name for log file (timestamp will be added)'
        )
        
        # Add subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Enrich command
        enrich_parser = subparsers.add_parser(
            'enrich',
            help='Enrich location data with standardized information'
        )
        self._add_enrich_arguments(enrich_parser)
        
        # Validate command
        validate_parser = subparsers.add_parser(
            'validate',
            help='Validate location data without enrichment'
        )
        self._add_validate_arguments(validate_parser)
        
        # Stats command
        stats_parser = subparsers.add_parser(
            'stats',
            help='Generate statistics about location data'
        )
        self._add_stats_arguments(stats_parser)
        
        return parser
    
    def _add_enrich_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for the enrich command."""
        # Required arguments
        parser.add_argument(
            '--input',
            type=str,
            required=True,
            help='Input CSV file path'
        )
        
        parser.add_argument(
            '--output',
            type=str,
            required=True,
            help='Output CSV file path'
        )
        
        # Optional arguments
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of records to process in each batch'
        )
        
        parser.add_argument(
            '--h3-resolution',
            type=int,
            choices=range(0, 16),
            default=9,
            help='H3 grid resolution (0-15)'
        )
        
        parser.add_argument(
            '--geometry-format',
            choices=['wkb', 'wkt', 'both'],
            default='both',
            help='Output geometry format'
        )
        
        parser.add_argument(
            '--error-handling',
            choices=['strict', 'warn', 'ignore'],
            default='warn',
            help='How to handle processing errors'
        )
        
        parser.add_argument(
            '--no-cache',
            action='store_true',
            help='Disable response caching'
        )
        
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
            '--progress-interval',
            type=int,
            default=1000,
            help='Number of records between progress updates'
        )
    
    def _add_validate_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for the validate command."""
        parser.add_argument(
            '--input',
            type=str,
            required=True,
            help='Input CSV file to validate'
        )
        
        parser.add_argument(
            '--report',
            type=str,
            help='Save validation report to file'
        )
        
        parser.add_argument(
            '--strict',
            action='store_true',
            help='Enable strict validation'
        )
    
    def _add_stats_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for the stats command."""
        parser.add_argument(
            '--input',
            type=str,
            required=True,
            help='Input CSV file to analyze'
        )
        
        parser.add_argument(
            '--output',
            type=str,
            help='Save statistics to file'
        )
        
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Generate detailed statistics'
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            console.print(f"[red]Error loading configuration file: {str(e)}")
            sys.exit(1)
    
    def _create_enrichment_config(self, args: argparse.Namespace) -> EnrichmentConfig:
        """Create enrichment configuration from arguments."""
        return EnrichmentConfig(
            input_file=Path(args.input),
            output_file=Path(args.output),
            batch_size=args.batch_size,
            h3_resolution=args.h3_resolution,
            validate_coordinates=not args.no_validate,
            enrich_addresses=not args.no_address,
            standardize_arabic=not args.no_arabic,
            add_uae_parsing=not args.no_uae,
            geometry_format=args.geometry_format,
            cache_responses=not args.no_cache,
            error_handling=args.error_handling,
            progress_log_interval=args.progress_interval
        )
    
    def enrich_command(self, args: argparse.Namespace) -> None:
        """Handle the enrich command."""
        try:
            config = self._create_enrichment_config(args)
            
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(
                    "[green]Enriching data...",
                    total=None
                )
                
                enricher = LocationDataEnricher(config)
                enricher.process_file()
                
                progress.update(task, completed=True)
            
            console.print("\n[green]Enrichment completed successfully!")
            
        except Exception as e:
            console.print(f"[red]Error during enrichment: {str(e)}")
            if args.log_level == 'DEBUG':
                console.print_exception()
            sys.exit(1)
    
    def validate_command(self, args: argparse.Namespace) -> None:
        """Handle the validate command."""
        try:
            # Implement validation logic here
            console.print("[yellow]Validation command not implemented yet")
            
        except Exception as e:
            console.print(f"[red]Error during validation: {str(e)}")
            sys.exit(1)
    
    def stats_command(self, args: argparse.Namespace) -> None:
        """Handle the stats command."""
        try:
            # Implement statistics logic here
            console.print("[yellow]Stats command not implemented yet")
            
        except Exception as e:
            console.print(f"[red]Error generating statistics: {str(e)}")
            sys.exit(1)
    
    def run(self) -> None:
        """Run the CLI application."""
        args = self.parser.parse_args()
        
        if args.config:
            config_data = self._load_config(args.config)
            # Update args with config file values
            for key, value in config_data.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
        
        # Setup logging
        setup_logging(args.log_level, args.log_file)
        
        # Handle commands
        if args.command == 'enrich':
            self.enrich_command(args)
        elif args.command == 'validate':
            self.validate_command(args)
        elif args.command == 'stats':
            self.stats_command(args)
        else:
            self.parser.print_help()
            sys.exit(1)

def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli = GeodataCLI()
        cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
