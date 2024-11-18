import requests
import csv
import time
import logging
from datetime import datetime
from math import cos, radians

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

url = "https://places.googleapis.com/v1/places:searchNearby"
headers = {
    "Content-Type": "application/json",
    "User-Agent": "insomnia/8.6.1",
    "X-Goog-Api-Key": "AIzaSyCb9bUxqmQpAfYdkla90LY3GQDcrjqme5Y",
    "X-Goog-FieldMask": "places.displayName,places.id,places.location,places.primaryType,places.primaryTypeDisplayName,places.types"
}

place_types = [
    "dental_clinic",
    "dentist",
    "doctor",
    "drugstore",
    "hospital",
    "medical_lab",
    "pharmacy",
    "physiotherapist"
]

# # Define the latitude and longitude bounds for the UAE
# lat_min, lat_max = 24.00, 26.05
# lng_min, lng_max = 51.43, 56.23

# Define the latitude and longitude bounds for the Sharjah
lat_min, lat_max = 24.781717, 25.519049
lng_min, lng_max = 55.349852, 56.374105

# Define the radius and step size for the search circles
radius = 2000  # in meters
step = radius  # move the center by the same amount as the radius

# Generate a unique filename based on the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"uae_places_{timestamp}.csv"

# Calculate the step size in degrees based on the radius and latitude
step_lat = step / 111111  # 1 degree latitude is approximately 111,111 meters
step_lng = step / (111111 * cos(radians(lat_min)))  # Adjust step size for longitude based on latitude

# Iterate over the latitude and longitude ranges in a zig-zag pattern
lat = lat_max
while lat >= lat_min:
    lng = lng_min
    while lng <= lng_max:
        # Iterate over the place types
        for place_type in place_types:
            payload = {
                "includedTypes": [place_type],
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": lat,
                            "longitude": lng
                        },
                        "radius": radius
                    }
                }
            }

            try:
                response = requests.post(url, json=payload, headers=headers)
                logging.info(f"Request - {place_type} - {lat}, {lng}")
                logging.info(f"Response status code: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    logging.info(f"Response content: {data}")

                    if "places" in data:
                        # Open the CSV file in append mode for each API call
                        with open(filename, mode="a", newline="", encoding='utf-8') as file:
                            writer = csv.writer(file)

                            # Write the header row if the file is empty
                            if file.tell() == 0:
                                writer.writerow(["displayName", "id", "latitude", "longitude", "primaryType", "primaryTypeDisplayName", "types"])

                            for place in data["places"]:
                                try:
                                    writer.writerow([
                                        place.get("displayName", {}).get("text", ""),
                                        place.get("id", ""),
                                        place.get("location", {}).get("latitude", ""),
                                        place.get("location", {}).get("longitude", ""),
                                        place.get("primaryType", ""),
                                        place.get("primaryTypeDisplayName", {}).get("text", ""),
                                        ",".join(place.get("types", []))
                                    ])
                                except KeyError as e:
                                    logging.warning(f"Missing key in place data: {e}")
                            logging.info(f"Data written to file for {place_type} - {lat}, {lng}")
                    else:
                        logging.warning(f"No places found in response for {place_type} - {lat}, {lng}")
                else:
                    logging.error(f"Error in API request: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                logging.error(f"Error in API request: {e}")

            time.sleep(0.2)

        lng += step_lng

    lat -= step_lat
    lng_max, lng_min = lng_min, lng_max  # Reverse the longitude direction for the next row

logging.info("Data collection completed.")