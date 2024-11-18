#!/usr/bin/env python3

import argparse
import logging
import sys
import os
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from contextlib import contextmanager

def setup_logging(log_level, log_file=None):
    """
    Setup logging configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler if log_file is specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Compare two large CSV files and identify differences using Dask.'
    )

    parser.add_argument(
        'file1',
        type=str,
        help='Path to the first CSV file.'
    )

    parser.add_argument(
        'file2',
        type=str,
        help='Path to the second CSV file.'
    )

    parser.add_argument(
        '-k', '--key-columns',
        type=str,
        required=True,
        nargs='+',
        help='Column(s) that uniquely identify each row (e.g., --key-columns id).'
    )

    parser.add_argument(
        '-u1', '--unique1',
        type=str,
        default='unique_to_file1.csv',
        help='Output CSV file for rows unique to the first file. Default: unique_to_file1.csv'
    )

    parser.add_argument(
        '-u2', '--unique2',
        type=str,
        default='unique_to_file2.csv',
        help='Output CSV file for rows unique to the second file. Default: unique_to_file2.csv'
    )

    parser.add_argument(
        '-d', '--differences',
        type=str,
        default='differences.csv',
        help='Output CSV file for rows with differing values. Default: differences.csv'
    )

    parser.add_argument(
        '-s', '--sample-size',
        type=int,
        default=1000000,
        help='Number of rows to read as a sample to infer data types. Default: 1,000,000'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level. Default: INFO'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to a file to store logs. If not set, logs are only printed to the console.'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar.'
    )

    parser.add_argument(
        '--scheduler',
        type=str,
        default='threads',
        choices=['threads', 'processes', 'single-threaded'],
        help='Dask scheduler to use. Default: threads'
    )

    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Number of workers for Dask scheduler. Defaults depend on scheduler type.'
    )

    return parser.parse_args()

@contextmanager
def nullcontext():
    """A no-op context manager."""
    yield

def initialize_dask_client(scheduler, n_workers):
    """
    Initialize Dask Client based on the scheduler type and number of workers.
    """
    try:
        if scheduler == 'threads':
            cluster = LocalCluster(
                n_workers=n_workers or os.cpu_count(),
                threads_per_worker=1,
                processes=True,
                dashboard_address=':8787'  # Optional: Specify dashboard port
            )
            logging.info(f"Initialized LocalCluster with threads scheduler: {cluster}")
        elif scheduler == 'processes':
            cluster = LocalCluster(
                n_workers=n_workers or os.cpu_count(),
                threads_per_worker=1,
                processes=True,
                dashboard_address=':8787'  # Optional
            )
            logging.info(f"Initialized LocalCluster with processes scheduler: {cluster}")
        elif scheduler == 'single-threaded':
            cluster = LocalCluster(
                n_workers=1,
                threads_per_worker=1,
                processes=False,
                dashboard_address=None  # No dashboard for single-threaded
            )
            logging.info(f"Initialized LocalCluster with single-threaded scheduler: {cluster}")
        else:
            logging.error(f"Unsupported scheduler type: {scheduler}")
            sys.exit(1)

        client = Client(cluster)
        logging.info(f"Dask LocalCluster initialized with {len(cluster.workers)} workers.")
        logging.info(f"Dask Dashboard available at {client.dashboard_link}")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Dask client: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during Dask client initialization: {e}")
        sys.exit(1)

def compare_large_csv(args):
    """
    Compare two large CSV files and identify differences.
    """

    # Initialize Dask client
    client = initialize_dask_client(args.scheduler, args.n_workers)

    try:
        # Check if input files exist
        if not os.path.isfile(args.file1):
            logging.error(f"File not found: {args.file1}")
            sys.exit(1)
        if not os.path.isfile(args.file2):
            logging.error(f"File not found: {args.file2}")
            sys.exit(1)

        # Read a sample to infer dtypes
        logging.info(f"Reading sample from {args.file1} to infer data types...")
        sample1 = pd.read_csv(args.file1, nrows=args.sample_size)
        dtypes = sample1.dtypes.to_dict()
        logging.debug(f"Inferred data types: {dtypes}")

        # Load CSVs with Dask
        logging.info(f"Loading {args.file1} with Dask...")
        df1 = dd.read_csv(args.file1, dtype=dtypes, assume_missing=True, low_memory=False)
        logging.info(f"Loading {args.file2} with Dask...")
        df2 = dd.read_csv(args.file2, dtype=dtypes, assume_missing=True, low_memory=False)

        # Validate key columns exist
        for col in args.key_columns:
            if col not in df1.columns:
                logging.error(f"Key column '{col}' not found in {args.file1}.")
                sys.exit(1)
            if col not in df2.columns:
                logging.error(f"Key column '{col}' not found in {args.file2}.")
                sys.exit(1)

        # Set index for efficient join
        logging.info(f"Setting index on key columns: {args.key_columns}")
        df1 = df1.set_index(args.key_columns, sorted=False, drop=True)
        df2 = df2.set_index(args.key_columns, sorted=False, drop=True)

        # Perform an outer join with indicator
        logging.info("Merging DataFrames to identify unique and common rows...")
        with ProgressBar() if not args.no_progress else nullcontext():
            merged = df1.merge(
                df2,
                how='outer',
                indicator=True,
                suffixes=('_file1', '_file2'),
                # on=args.key_columns  # Not needed since we set the index
            )

        # Rows unique to file1
        logging.info("Identifying rows unique to the first file...")
        unique_to_file1 = merged[merged['_merge'] == 'left_only']
        unique1_count = unique_to_file1.shape[0].compute()
        if unique1_count > 0:
            logging.info(f"Exporting {unique1_count} unique rows to {args.unique1}")
            unique_to_file1 = unique_to_file1.reset_index()
            unique_to_file1.to_csv(args.unique1, single_file=True, index=False)
        else:
            logging.info(f"No unique rows found in {args.file1}")

        # Rows unique to file2
        logging.info("Identifying rows unique to the second file...")
        unique_to_file2 = merged[merged['_merge'] == 'right_only']
        unique2_count = unique_to_file2.shape[0].compute()
        if unique2_count > 0:
            logging.info(f"Exporting {unique2_count} unique rows to {args.unique2}")
            unique_to_file2 = unique_to_file2.reset_index()
            unique_to_file2.to_csv(args.unique2, single_file=True, index=False)
        else:
            logging.info(f"No unique rows found in {args.file2}")

        # Rows present in both but with differences
        logging.info("Identifying rows with differing values...")
        common = merged[merged['_merge'] == 'both']

        if common.shape[0].compute() == 0:
            logging.info("No common rows to compare for differences.")
            return

        # Identify differing rows
        compare_columns = [col for col in df1.columns if col not in args.key_columns + ['_merge']]
        logging.debug(f"Columns to compare for differences: {compare_columns}")

        # Build a boolean mask where any of the compared columns differ
        logging.info("Building mask for differing rows...")
        diff_mask = None
        for col in compare_columns:
            col_file1 = f"{col}_file1"
            col_file2 = f"{col}_file2"
            if col_file1 in common.columns and col_file2 in common.columns:
                if diff_mask is None:
                    diff_mask = common[col_file1] != common[col_file2]
                else:
                    diff_mask |= common[col_file1] != common[col_file2]
            else:
                logging.warning(f"Column pair {col_file1}, {col_file2} not found in merged DataFrame.")

        if diff_mask is None:
            logging.warning("No columns available to compare for differences.")
            differing_rows = dd.from_pandas(pd.DataFrame(), npartitions=1)
        else:
            differing_rows = common[diff_mask]

        differing_count = differing_rows.shape[0].compute()
        if differing_count > 0:
            logging.info(f"Exporting {differing_count} differing rows to {args.differences}")
            differing_rows = differing_rows.reset_index()
            differing_rows.to_csv(args.differences, single_file=True, index=False)
        else:
            logging.info("No differing rows found between the files.")

    except pd.errors.EmptyDataError as ede:
        logging.error(f"Empty data error: {ede}")
        sys.exit(1)
    except pd.errors.ParserError as pe:
        logging.error(f"Parser error: {pe}")
        sys.exit(1)
    except FileNotFoundError as fnfe:
        logging.error(f"File not found: {fnfe}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        client.close()
        logging.info("Dask client closed.")

def main():
    args = parse_arguments()
    setup_logging(args.log_level, args.log_file)

    logging.debug(f"Parsed arguments: {args}")
    compare_large_csv(args)

if __name__ == "__main__":
    main()
