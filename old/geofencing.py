import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, box
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_polygon_coordinates(kml_file):
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        
        # Find the <coordinates> tag within the KML file
        coordinates_elem = root.find('.//{http://www.opengis.net/kml/2.2}coordinates')
        
        if coordinates_elem is not None:
            coordinates = coordinates_elem.text.strip().split()
            polygon_coords = [(float(coord.split(',')[0]), float(coord.split(',')[1])) for coord in coordinates]
            logging.info(f"Extracted {len(polygon_coords)} polygon coordinates from the KML file.")
            return polygon_coords
        else:
            logging.warning("No <coordinates> tag found in the KML file.")
            return None
    except ET.ParseError as e:
        logging.error(f"Error parsing the KML file: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while extracting polygon coordinates: {e}")
        return None

def create_geofencing_rectangles(polygon_coords, size=2):
    polygon = Polygon(polygon_coords)
    min_x, min_y, max_x, max_y = polygon.bounds
    
    rectangles = []
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            rectangle = box(x, y, x + size, y + size)
            if rectangle.intersects(polygon):
                while rectangle.union(box(rectangle.bounds[0], rectangle.bounds[1], rectangle.bounds[2], rectangle.bounds[3] + size)).intersects(polygon):
                    rectangle = rectangle.union(box(rectangle.bounds[0], rectangle.bounds[1], rectangle.bounds[2], rectangle.bounds[3] + size))
                while rectangle.union(box(rectangle.bounds[0], rectangle.bounds[1], rectangle.bounds[2] + size, rectangle.bounds[3])).intersects(polygon):
                    rectangle = rectangle.union(box(rectangle.bounds[0], rectangle.bounds[1], rectangle.bounds[2] + size, rectangle.bounds[3]))
                rectangles.append(rectangle)
            y += size
        x += size
    
    logging.info(f"Created {len(rectangles)} geofencing rectangles.")
    return rectangles

def merge_squares(squares):
    merged_rectangles = []
    
    while squares:
        current_square = squares.pop(0)
        merged_rectangle = current_square
        
        i = 0
        while i < len(squares):
            if merged_rectangle.union(squares[i]).area == merged_rectangle.area + squares[i].area:
                merged_rectangle = merged_rectangle.union(squares[i])
                squares.pop(i)
            else:
                i += 1
        
        merged_rectangles.append(merged_rectangle)
    
    logging.info(f"Merged {len(squares)} squares into {len(merged_rectangles)} rectangles.")
    return merged_rectangles

def visualize_geofencing_rectangles(polygon, rectangles):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the polygon
    x, y = polygon.exterior.xy
    ax.plot(x, y, color='blue', linewidth=2, label='Polygon')
    
    # Plot the geofencing rectangles
    for rectangle in rectangles:
        x, y = rectangle.exterior.xy
        ax.plot(x, y, color='red', linewidth=1, label='Rectangle')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geofencing Rectangles')
    ax.legend()
    plt.tight_layout()
    plt.show()

def save_rectangles_to_file(rectangles, output_file):
    with open(output_file, 'w') as file:
        for i, rectangle in enumerate(rectangles, start=1):
            nw_point = (rectangle.bounds[0], rectangle.bounds[3])
            se_point = (rectangle.bounds[2], rectangle.bounds[1])
            file.write(f"Rectangle {i}: NW Point: {nw_point}, SE Point: {se_point}\n")
    logging.info(f"Saved {len(rectangles)} rectangles to {output_file}.")

def main(kml_file, output_file):
    polygon_coords = extract_polygon_coordinates(kml_file)
    
    if polygon_coords:
        polygon = Polygon(polygon_coords)
        rectangles = create_geofencing_rectangles(polygon_coords)
        
        for i, rectangle in enumerate(rectangles, start=1):
            nw_point = (rectangle.bounds[0], rectangle.bounds[3])
            se_point = (rectangle.bounds[2], rectangle.bounds[1])
            logging.info(f"Rectangle {i}: NW Point: {nw_point}, SE Point: {se_point}")
        
        visualize_geofencing_rectangles(polygon, rectangles)
        save_rectangles_to_file(rectangles, output_file)
    else:
        logging.warning("No polygon coordinates found in the KML file.")

# Usage example
kml_file = 'polygon.kml'
output_file = 'geofencing_rectangles.txt'
try:
    main(kml_file, output_file)
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")