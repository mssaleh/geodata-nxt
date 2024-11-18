import pandas as pd
import os
import argparse
import logging
from typing import List, Optional
from datetime import datetime

class ExcelMerger:
    def __init__(self, input_folder: str, output_file: str, remove_duplicates: bool = False, 
                 unique_field: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize the Excel Merger with the given parameters.
        
        Args:
            input_folder (str): Path to the folder containing Excel files
            output_file (str): Path for the output Excel file
            remove_duplicates (bool): Whether to remove duplicate rows
            unique_field (str, optional): Column name for identifying duplicates
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.input_folder = input_folder
        self.output_file = output_file
        self.remove_duplicates = remove_duplicates
        self.unique_field = unique_field
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Validate initialization parameters
        self._validate_params()

    def setup_logging(self, log_level: str) -> None:
        """Set up logging configuration."""
        log_filename = f"excel_merger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _validate_params(self) -> None:
        """Validate input parameters."""
        if not os.path.exists(self.input_folder):
            raise ValueError(f"Input folder does not exist: {self.input_folder}")
        
        if not os.path.isdir(self.input_folder):
            raise ValueError(f"Input path is not a directory: {self.input_folder}")
        
        output_dir = os.path.dirname(os.path.abspath(self.output_file))
        if not os.path.exists(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")
        
        if self.remove_duplicates and not self.unique_field:
            raise ValueError("unique_field must be specified when remove_duplicates is True")

    def get_excel_files(self) -> List[str]:
        """Get list of Excel files in the input folder."""
        excel_files = [f for f in os.listdir(self.input_folder) 
                      if f.endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            raise FileNotFoundError(f"No Excel files found in {self.input_folder}")
        
        return excel_files

    def read_excel_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Read a single Excel file and return its contents as a DataFrame.
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            Optional[pd.DataFrame]: DataFrame if successful, None if failed
        """
        try:
            df = pd.read_excel(file_path)
            if df.empty:
                self.logger.warning(f"File is empty: {file_path}")
                return None
            return df
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    def validate_dataframe(self, df: pd.DataFrame, file_path: str) -> bool:
        """
        Validate DataFrame structure and content.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            file_path (str): Path to the source file (for logging)
            
        Returns:
            bool: True if valid, False otherwise
        """
        if self.remove_duplicates and self.unique_field not in df.columns:
            self.logger.error(
                f"Unique field '{self.unique_field}' not found in file: {file_path}"
            )
            return False
        return True

    def merge_excel_files(self) -> None:
        """Merge all Excel files in the input folder into a single file."""
        try:
            self.logger.info("Starting Excel merge process")
            excel_files = self.get_excel_files()
            self.logger.info(f"Found {len(excel_files)} Excel files")

            all_dataframes = []
            
            # Read and process each Excel file
            for file in excel_files:
                file_path = os.path.join(self.input_folder, file)
                self.logger.debug(f"Processing file: {file}")
                
                df = self.read_excel_file(file_path)
                if df is None:
                    continue
                    
                if not self.validate_dataframe(df, file_path):
                    continue
                    
                all_dataframes.append(df)
                self.logger.info(f"Successfully read: {file} ({len(df)} rows)")

            if not all_dataframes:
                raise ValueError("No valid data was read from any Excel file")

            # Combine all dataframes
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            self.logger.info(f"Combined data: {len(combined_df)} rows")

            # Remove duplicates if requested
            if self.remove_duplicates:
                initial_rows = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=[self.unique_field], keep='first')
                removed_rows = initial_rows - len(combined_df)
                self.logger.info(
                    f"Removed {removed_rows} duplicate rows based on '{self.unique_field}'"
                )

            # Save to output file
            combined_df.to_excel(self.output_file, index=False)
            self.logger.info(
                f"Merged data saved to: {self.output_file} "
                f"(Total rows: {len(combined_df)})"
            )

        except Exception as e:
            self.logger.error(f"An error occurred during the merge process: {str(e)}")
            raise

def main():
    """Main entry point for the script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Merge multiple Excel files into a single file"
    )
    parser.add_argument(
        "-i", "--input-folder",
        required=True,
        help="Input folder containing Excel files"
    )
    parser.add_argument(
        "-o", "--output-file",
        required=True,
        help="Output Excel file path"
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Remove duplicate rows based on a unique field"
    )
    parser.add_argument(
        "--field-name",
        help="Name of the column containing unique identifiers"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )

    args = parser.parse_args()

    try:
        merger = ExcelMerger(
            input_folder=args.input_folder,
            output_file=args.output_file,
            remove_duplicates=args.remove_duplicates,
            unique_field=args.field_name,
            log_level=args.log_level
        )
        merger.merge_excel_files()
    except Exception as e:
        logging.error(f"Script execution failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
