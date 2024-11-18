import csv
import sys
import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Input CSV file path containing geolocations (id, latitude, longitude)
input_file = 'geolocations.csv'

# Output CSV file path for ISO 3166-2 codes
output_file = 'iso_codes.csv'

# Function to retrieve the ISO 3166-2 code using OpenStreetMap Nominatim API
def get_iso_code(lat, lng):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&zoom=8"
    headers = {
        'User-Agent': 'YourAppName/1.0'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        data = response.json()
        if 'address' in data:
            address = data['address']
            if 'country_code' in address and 'state' in address:
                return f"{address['country_code'].upper()}-{address['state']}"
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"OpenStreetMap Nominatim API request failed. {str(e)}")
        return None

# Get the total number of rows in the input CSV file
try:
    with open(input_file, 'r', encoding='utf-8') as input_csvfile:
        reader = csv.reader(input_csvfile)
        total_rows = sum(1 for _ in reader) - 1  # Subtract 1 for the header row
except FileNotFoundError:
    logging.error(f"Input CSV file not found: {input_file}")
    sys.exit(1)
except csv.Error as e:
    logging.error(f"Failed to read input CSV file: {input_file}. {str(e)}")
    sys.exit(1)

# Open the output CSV file in append mode
try:
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if the file is empty
        csvfile.seek(0)
        if csvfile.tell() == 0:
            writer.writerow(['ID', 'Latitude', 'Longitude', 'ISO 3166-2 Code'])

        # Read geolocations from the input CSV file and process each row
        try:
            with open(input_file, 'r', encoding='utf-8') as input_csvfile:
                reader = csv.reader(input_csvfile)
                next(reader, None)  # Skip the header row if present

                for index, row in enumerate(reader, start=1):
                    if len(row) >= 3:
                        id, lat, lng = row[0], row[1], row[2]
                        iso_code = get_iso_code(lat, lng)
                        time.sleep(1)  # Rate-limit OpenStreetMap API requests to 1 per second
                        writer.writerow([id, lat, lng, iso_code])  # Write the ISO code to the output file
                        logging.info(f"Processed row {index} of {total_rows}")
                    else:
                        logging.warning(f"Invalid row format at line {index} in input CSV file.")
        except csv.Error as e:
            logging.error(f"Failed to read input CSV file: {input_file}. {str(e)}")
            sys.exit(1)

    logging.info(f"ISO 3166-2 codes exported to {output_file}")
except IOError as e:
    logging.error(f"Failed to write output CSV file: {output_file}. {str(e)}")
    sys.exit(1)