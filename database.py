import psycopg2
from psycopg2 import Error
import psycopg2.extras
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration from environment variables
DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME")
}

def sanitize_table_name(filename):
    """Convert filename to a valid PostgreSQL table name (exact filename without extension)."""
    # Remove extension only
    name = filename.replace('.tif', '')
    # Keep original case and characters - we'll use quoted identifiers
    return name

def drop_old_tables():
    """Drop the old table structure if it exists."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS predictions CASCADE")
        cur.execute("DROP TABLE IF EXISTS detections CASCADE")
        conn.commit()
        cur.close()
        print("Old tables dropped successfully.")
    except (Exception, Error) as error:
        print(f"Error dropping old tables: {error}")
    finally:
        if conn is not None:
            conn.close()

def create_table_for_image(filename):
    """
    Creates a dynamic table for the given image filename.
    Table schema stores GEOGRAPHIC COORDINATES (latitude/longitude):
    - id: TEXT (format: objectname_1, objectname_2, etc.)
    - object_name: TEXT
    - top_left_lat, top_left_lon: REAL (geographic coordinates)
    - bottom_right_lat, bottom_right_lon: REAL (geographic coordinates)
    - centroid_lat, centroid_lon: REAL (center point in lat/lon)
    - cropped_image_path: TEXT
    - accuracy_percentage: REAL
    """
    table_name = sanitize_table_name(filename)
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Use quoted identifier to allow exact filename as table name
        # Schema stores GEOGRAPHIC COORDINATES (lat/lon)
        create_command = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id TEXT PRIMARY KEY,
            object_name TEXT NOT NULL,
            top_left_lat REAL NOT NULL,
            top_left_lon REAL NOT NULL,
            bottom_right_lat REAL NOT NULL,
            bottom_right_lon REAL NOT NULL,
            centroid_lat REAL NOT NULL,
            centroid_lon REAL NOT NULL,
            cropped_image_path TEXT NOT NULL,
            accuracy_percentage REAL NOT NULL
        )
        """
        cur.execute(create_command)
        conn.commit()
        cur.close()
        print(f"Table '{table_name}' created/verified.")
        return table_name
    except (Exception, Error) as error:
        print(f"Error creating table for {filename}: {error}")
        return None
    finally:
        if conn is not None:
            conn.close()

def save_object_to_table(table_name, object_id, object_name, geo_coords, cropped_path, accuracy):
    """
    Saves a single detected object to the specified table with GEOGRAPHIC coordinates.
    
    Args:
        table_name: Name of the table (sanitized filename)
        object_id: Unique ID (e.g., 'ship_1')
        object_name: Category name (e.g., 'ship')
        geo_coords: Dict with keys: 'top_left_lat', 'top_left_lon', 'bottom_right_lat', 'bottom_right_lon', 'centroid_lat', 'centroid_lon'
        cropped_path: Path to the cropped image
        accuracy: Accuracy percentage
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Use quoted identifier for table name
        insert_command = f"""
        INSERT INTO "{table_name}" (
            id, object_name, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon,
            centroid_lat, centroid_lon, cropped_image_path, accuracy_percentage
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cur.execute(insert_command, (
            object_id,
            object_name,
            geo_coords['top_left_lat'],
            geo_coords['top_left_lon'],
            geo_coords['bottom_right_lat'],
            geo_coords['bottom_right_lon'],
            geo_coords['centroid_lat'],
            geo_coords['centroid_lon'],
            cropped_path,
            accuracy
        ))
        
        conn.commit()
        cur.close()
    except (Exception, Error) as error:
        print(f"Error saving object {object_id}: {error}")
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    # Test: drop old tables
    drop_old_tables()
    print("Database restructured. Ready for dynamic table creation with geographic coordinates.")
