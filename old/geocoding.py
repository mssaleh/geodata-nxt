import googlemaps
import csv
import sys
import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Google Maps client with your API key
try:
    gmaps = googlemaps.Client(key='AIzaSyCb9bUxqmQpAfYdkla90LY3GQDcrjqme5Y')
except ValueError as e:
    logging.error(f"Invalid Google Maps API key. {str(e)}")
    sys.exit(1)

# Input CSV file path containing place IDs
input_file = 'place_ids.csv'

# Output CSV file path for geocoded data
output_file = 'geocoded_data.csv'

# Function to geocode a place ID and return the coordinates
def geocode_place_id(place_id):
    try:
        result = gmaps.geocode(place_id=place_id)
        if result:
            location = result[0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            logging.warning(f"No results found for place ID: {place_id}")
            return None, None
    except googlemaps.exceptions.ApiError as e:
        logging.error(f"Geocoding API request failed for place ID: {place_id}. {str(e)}")
        return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred for place ID: {place_id}. {str(e)}")
        return None, None

# Open the output CSV file in append mode
try:
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header row if the file is empty
        csvfile.seek(0)
        if csvfile.tell() == 0:
            writer.writerow(['Place ID', 'Latitude', 'Longitude'])
        
        # Read place IDs from the input CSV file and process each row
        try:
            with open(input_file, 'r', encoding='utf-8') as input_csvfile:
                reader = csv.reader(input_csvfile)
                next(reader, None)  # Skip the header row if present
                total_rows = sum(1 for _ in reader)
                input_csvfile.seek(0)  # Reset the file pointer
                next(reader, None)  # Skip the header row if present
                
                for index, row in enumerate(reader, start=1):
                    if len(row) > 0:
                        place_id = row[0]
                        lat, lng = geocode_place_id(place_id)
                        time.sleep(0.5)  # Rate-limit OpenStreetMap API requests to 2 per second
                        writer.writerow([place_id, lat, lng])  # Write the geocoded data to the output file
                        logging.info(f"Processed row {index} of {total_rows}")
                    else:
                        logging.warning(f"Empty row encountered in input CSV file.")
        except FileNotFoundError:
            logging.error(f"Input CSV file not found: {input_file}")
            sys.exit(1)
        except csv.Error as e:
            logging.error(f"Failed to read input CSV file: {input_file}. {str(e)}")
            sys.exit(1)
        
    logging.info(f"Geocoded data exported to {output_file}")
except IOError as e:
    logging.error(f"Failed to write output CSV file: {output_file}. {str(e)}")
    sys.exit(1)