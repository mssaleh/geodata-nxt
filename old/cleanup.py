import dask.dataframe as dd
import pandas as pd
import re
import unicodedata
import logging
import sys
from dask.diagnostics import ProgressBar
import argparse
from dask.distributed import Client, LocalCluster

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_cleaning.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def clean_line_terminators(text):
    """
    Replace Unicode Line Separator and Paragraph Separator with a space.
    """
    if pd.isnull(text):
        return text
    return re.sub(r'[\u2028\u2029]', ' ', text)

def remove_control_characters(text):
    """
    Remove all Unicode control characters except common punctuation.
    """
    if pd.isnull(text):
        return text
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

def remove_emojis(text):
    """
    Remove emojis and other pictographic symbols from text.
    """
    if pd.isnull(text):
        return text
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def normalize_unicode_text(text):
    """
    Normalize Unicode text to NFC (Normalization Form C).
    """
    if pd.isnull(text):
        return text
    return unicodedata.normalize('NFC', text)

def remove_extra_whitespace(text):
    """
    Remove extra whitespace from text.
    """
    if pd.isnull(text):
        return text
    return ' '.join(text.split())

def convert_to_lowercase(text):
    """
    Convert text to lowercase.
    """
    if pd.isnull(text):
        return text
    return text.lower()

def remove_specific_patterns(text):
    """
    Remove specific patterns from text, such as URLs.
    """
    if pd.isnull(text):
        return text
    # Example: Remove URLs
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_urls_vectorized(df, text_columns):
    """
    Remove URLs from specified text columns using vectorized operations.
    
    Parameters:
    - df (dask.dataframe.DataFrame): The Dask DataFrame.
    - text_columns (list): Columns to apply the function on.
    
    Returns:
    - dask.dataframe.DataFrame: The DataFrame with URLs removed.
    """
    url_pattern = r'http\S+|www\.\S+'
    for col in text_columns:
        logging.debug(f"Vectorized removing URLs from column: {col}")
        df[col] = df[col].str.replace(url_pattern, '', regex=True)
    return df

def convert_to_lowercase_vectorized(df, text_columns):
    """
    Convert text columns to lowercase using Dask's string methods.
    
    Parameters:
    - df (dask.dataframe.DataFrame): The Dask DataFrame.
    - text_columns (list): Columns to apply the function on.
    
    Returns:
    - dask.dataframe.DataFrame: The DataFrame with lowercase text.
    """
    for col in text_columns:
        logging.debug(f"Vectorized converting to lowercase for column: {col}")
        df[col] = df[col].str.lower()
    return df

def apply_cleaning_function(df_partition, cleaning_func, text_columns):
    """
    Apply a cleaning function to specified text columns within a partition.
    
    Parameters:
    - df_partition (pd.DataFrame): The partition of the DataFrame.
    - cleaning_func (function): The cleaning function to apply.
    - text_columns (list): Columns to apply the function on.
    
    Returns:
    - pd.DataFrame: The cleaned partition.
    """
    for col in text_columns:
        try:
            original_length = len(df_partition[col])
            # Use .loc to avoid SettingWithCopyWarning
            df_partition.loc[:, col] = df_partition[col].apply(cleaning_func)
            new_length = len(df_partition[col])
            logging.debug(f"Cleaned column: {col}. Rows before: {original_length}, after: {new_length}")
        except Exception as e:
            logging.error(f"Error applying {cleaning_func.__name__} to column: {col}. Error: {e}")
            df_partition.loc[:, col] = df_partition[col]  # Retain original data in case of error
    return df_partition

def initialize_dask_cluster(n_workers=None, threads_per_worker=None, memory_limit='0'):
    """
    Initialize a Dask LocalCluster and return a Client instance.
    
    Parameters:
    - n_workers (int): Number of worker processes. Defaults to the number of CPU cores.
    - threads_per_worker (int): Number of threads per worker. Defaults to 1.
    - memory_limit (str): Memory limit per worker (e.g., '2GB'). '0' means no limit.
    
    Returns:
    - client (dask.distributed.Client): Dask client connected to the cluster.
    """
    try:
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
        client = Client(cluster)
        logging.info(f"Dask LocalCluster initialized with {len(cluster.workers)} workers and {threads_per_worker} threads per worker.")
        logging.info(f"Dask Dashboard available at {client.dashboard_link}")
        return client
    except Exception as e:
        logging.error("Failed to initialize Dask LocalCluster.", exc_info=True)
        sys.exit(1)

def deduplicate_dataframe(df, dedup_field):
    """
    Deduplicate the DataFrame based on the specified field.
    """
    if dedup_field not in df.columns:
        logging.error(f"Deduplication field '{dedup_field}' not found in DataFrame columns.")
        sys.exit(1)
    logging.info(f"Deduplicating DataFrame based on field: '{dedup_field}'")
    initial_count = len(df)
    df = df.drop_duplicates(subset=[dedup_field])
    final_count = len(df)
    logging.info(f"Deduplication complete. Rows before: {initial_count}, after: {final_count}.")
    return df

def clean_csv(input_file, output_file, remove_emojis_flag=False, specified_columns=None, deduplicate_flag=False, dedup_field=None,
              clean_line_terminators_flag=False, remove_control_characters_flag=False, normalize_unicode_flag=False,
              remove_extra_whitespace_flag=False, convert_to_lowercase_flag=False, remove_specific_patterns_flag=False):
    """
    Clean the CSV file by handling line terminators, control characters,
    emojis, Unicode normalization, whitespace, case conversion, and specific patterns.
    Optionally deduplicate the DataFrame based on a specified field.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the cleaned CSV file.
    - remove_emojis_flag (bool): Whether to remove emojis from text.
    - specified_columns (list): List of specific columns to clean. If None, all object dtype columns are cleaned.
    - deduplicate_flag (bool): Whether to perform deduplication.
    - dedup_field (str): The field name to deduplicate on.
    - clean_line_terminators_flag (bool): Whether to replace line terminators.
    - remove_control_characters_flag (bool): Whether to remove control characters.
    - normalize_unicode_flag (bool): Whether to normalize Unicode text.
    - remove_extra_whitespace_flag (bool): Whether to remove extra whitespace.
    - convert_to_lowercase_flag (bool): Whether to convert text to lowercase.
    - remove_specific_patterns_flag (bool): Whether to remove specific patterns like URLs.
    - n_workers (int): Number of Dask workers. Defaults to number of CPU cores.
    - threads_per_worker (int): Number of threads per Dask worker. Defaults to 1.
    - memory_limit (str): Memory limit per Dask worker (e.g., '2GB'). '0' means no limit.
    """
    
    try:
        logging.info(f"Starting processing for input file: {input_file}")
    
        # Read the CSV with Dask
        logging.info("Reading the CSV file with Dask")
        df = dd.read_csv(input_file, assume_missing=True, dtype='object', encoding='utf-8', blocksize='64MB')

        # Deduplication step (optional)
        if deduplicate_flag:
            if not dedup_field:
                logging.error("Deduplication field must be specified when deduplication is enabled.")
                sys.exit(1)
            logging.info("Performing deduplication")
            df = deduplicate_dataframe(df, dedup_field)
    
        # Identify text columns
        if specified_columns:
            # Verify specified columns exist
            existing_columns = [col for col in specified_columns if col in df.columns]
            missing_columns = set(specified_columns) - set(existing_columns)
            if missing_columns:
                logging.warning(f"The following specified columns are missing and will be skipped: {missing_columns}")
            text_columns = existing_columns
        else:
            # Automatically identify all object dtype columns
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
        logging.info(f"Columns identified for cleaning: {text_columns}")
    
        if not text_columns:
            logging.warning("No text columns found for cleaning.")
        else:
            # Define a list of tuples (function_name, function_reference, flag)
            cleaning_steps = [
                ('Clean Line Terminators', clean_line_terminators, clean_line_terminators_flag),
                ('Remove Control Characters', remove_control_characters, remove_control_characters_flag),
                ('Normalize Unicode Text', normalize_unicode_text, normalize_unicode_flag),
                ('Remove Extra Whitespace', remove_extra_whitespace, remove_extra_whitespace_flag),
                ('Convert to Lowercase', convert_to_lowercase, convert_to_lowercase_flag),
                ('Remove Specific Patterns', remove_specific_patterns, remove_specific_patterns_flag)
            ]
    
            # Conditionally add emoji removal
            if remove_emojis_flag:
                cleaning_steps.insert(2, ('Remove Emojis', remove_emojis, remove_emojis_flag))  # Insert after control characters
    
            # Apply cleaning functions to each text column
            for func_name, func, flag in cleaning_steps:
                if flag:
                    logging.info(f"Applying '{func_name}' to text columns")
                    if func_name == 'Remove Specific Patterns':
                        # Use vectorized operations for efficiency
                        df = remove_urls_vectorized(df, text_columns)
                    elif func_name == 'Convert to Lowercase':
                        # Use vectorized string methods
                        df = convert_to_lowercase_vectorized(df, text_columns)
                    elif func_name == 'Remove Emojis':
                        # Apply using map_partitions
                        df = df.map_partitions(apply_cleaning_function, cleaning_func=func, text_columns=text_columns, meta=df._meta)
                    else:
                        # Apply function using map_partitions for parallel processing
                        df = df.map_partitions(apply_cleaning_function, cleaning_func=func, text_columns=text_columns, meta=df._meta)
                else:
                    logging.info(f"Skipping '{func_name}' as it was not enabled.")
    
        # Write the cleaned data to a new CSV
        logging.info(f"Writing the cleaned data to output file: {output_file}")
        with ProgressBar():
            df.to_csv(output_file, single_file=True, index=False, encoding='utf-8')
    
        logging.info("Data cleaning complete.")

    except Exception as e:
        logging.error("An error occurred during the cleaning process.", exc_info=True)
        sys.exit(1)
    
def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Clean and optionally deduplicate a large CSV file by handling unusual characters and patterns.")

    # Input and output files
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input CSV file.'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save the cleaned CSV file.'
    )

    # Optional cleaning function flags
    parser.add_argument(
        '--clean-line-terminators',
        action='store_true',
        help='Enable replacing Unicode line terminators with a space.'
    )

    parser.add_argument(
        '--remove-control-characters',
        action='store_true',
        help='Enable removing non-printable control characters.'
    )

    parser.add_argument(
        '--remove-emojis',
        action='store_true',
        help='Enable removing emojis and pictographic symbols from text columns.'
    )

    parser.add_argument(
        '--normalize-unicode',
        action='store_true',
        help='Enable normalizing Unicode text to NFC form.'
    )

    parser.add_argument(
        '--remove-extra-whitespace',
        action='store_true',
        help='Enable removing excessive whitespace.'
    )

    parser.add_argument(
        '--convert-to-lowercase',
        action='store_true',
        help='Enable converting text to lowercase.'
    )

    parser.add_argument(
        '--remove-specific-patterns',
        action='store_true',
        help='Enable removing specific patterns like URLs.'
    )

    # Specify columns to clean
    parser.add_argument(
        '--columns',
        nargs='+',
        help='List of specific columns to clean. If not provided, all object dtype columns will be cleaned.'
    )

    # Deduplication flags
    parser.add_argument(
        '--deduplicate',
        action='store_true',
        help='Flag to enable deduplication of the CSV based on a specified field.'
    )

    parser.add_argument(
        '--dedup-field',
        type=str,
        help='The field name to deduplicate on. Required if --deduplicate is set.'
    )

    # Dask cluster configuration arguments (optional)
    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Number of Dask workers. Defaults to number of CPU cores.'
    )

    parser.add_argument(
        '--threads-per-worker',
        type=int,
        default=1,
        help='Number of threads per Dask worker. Defaults to 1.'
    )

    parser.add_argument(
        '--memory-limit',
        type=str,
        default='0',  # '0' means no limit
        help='Memory limit per Dask worker (e.g., "2GB"). Defaults to no limit.'
    )

    args = parser.parse_args()

    # Validate deduplication arguments
    if args.deduplicate and not args.dedup_field:
        logging.error("The --dedup-field argument must be specified when --deduplicate is set.")
        sys.exit(1)

    # Initialize Dask cluster
    client = initialize_dask_cluster(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker, memory_limit=args.memory_limit)

    # Log Dask cluster details
    logging.info(f"Active Dask Client: {client}")
    logging.info(f"Number of Workers: {len(client.scheduler_info()['workers'])}")
    for worker, info in client.scheduler_info()['workers'].items():
        logging.info(f"Worker: {worker}, Host: {info['host']}, Available Memory: {info['memory_limit']} bytes")

    # Call the cleaning function with parsed arguments
    clean_csv(
        input_file=args.input,
        output_file=args.output,
        remove_emojis_flag=args.remove_emojis,
        specified_columns=args.columns,
        deduplicate_flag=args.deduplicate,
        dedup_field=args.dedup_field,
        clean_line_terminators_flag=args.clean_line_terminators,
        remove_control_characters_flag=args.remove_control_characters,
        normalize_unicode_flag=args.normalize_unicode,
        remove_extra_whitespace_flag=args.remove_extra_whitespace,
        convert_to_lowercase_flag=args.convert_to_lowercase,
        remove_specific_patterns_flag=args.remove_specific_patterns
    )

    # Gracefully shutdown the Dask client
    client.close()
    logging.info("Dask Client shutdown gracefully.")

if __name__ == "__main__":
    main()
