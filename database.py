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

def create_change_reports_table():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        create_command = """
        CREATE TABLE IF NOT EXISTS change_reports (
            id SERIAL PRIMARY KEY,
            location_name TEXT NOT NULL,
            before_image TEXT NOT NULL,
            after_image TEXT NOT NULL,
            before_timestamp TIMESTAMP NOT NULL,
            after_timestamp TIMESTAMP NOT NULL,
            analysis_timestamp TIMESTAMP NOT NULL,
            total_slices INTEGER NOT NULL,
            slices_with_changes INTEGER NOT NULL,
            construction_detected BOOLEAN NOT NULL,
            construction_summary TEXT,
            vehicle_summary TEXT,
            object_changes_summary TEXT,
            critical_issues TEXT,
            detailed_report_path TEXT NOT NULL,
            overall_status TEXT NOT NULL
        )
        """
        cur.execute(create_command)
        conn.commit()
        cur.close()
        print("Table 'change_reports' created/verified.")
    except (Exception, Error) as error:
        print(f"Error creating change_reports table: {error}")
    finally:
        if conn is not None:
            conn.close()

def save_change_report(report_data):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        insert_command = """
        INSERT INTO change_reports (
            location_name, before_image, after_image, before_timestamp, after_timestamp,
            analysis_timestamp, total_slices, slices_with_changes, construction_detected,
            construction_summary, vehicle_summary, object_changes_summary, critical_issues,
            detailed_report_path, overall_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        cur.execute(insert_command, (
            report_data['location_name'],
            report_data['before_image'],
            report_data['after_image'],
            report_data['before_timestamp'],
            report_data['after_timestamp'],
            report_data['analysis_timestamp'],
            report_data['total_slices'],
            report_data['slices_with_changes'],
            report_data['construction_detected'],
            report_data['construction_summary'],
            report_data['vehicle_summary'],
            report_data['object_changes_summary'],
            report_data['critical_issues'],
            report_data['detailed_report_path'],
            report_data['overall_status']
        ))
        
        report_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        print(f"Change report saved to database with ID: {report_id}")
        return report_id
    except (Exception, Error) as error:
        print(f"Error saving change report: {error}")
        return None
    finally:
        if conn is not None:
            conn.close()

def query_objects_in_bounds(table_name, min_lat, max_lat, min_lon, max_lon):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = f"""
        SELECT * FROM "{table_name}"
        WHERE centroid_lat BETWEEN %s AND %s
        AND centroid_lon BETWEEN %s AND %s
        """
        
        cur.execute(query, (min_lat, max_lat, min_lon, max_lon))
        objects = cur.fetchall()
        cur.close()
        return [dict(obj) for obj in objects]
    except (Exception, Error) as error:
        print(f"Error querying objects from {table_name}: {error}")
        return []
    finally:
        if conn is not None:
            conn.close()

def create_change_analysis_table():
    """Create the change_analysis table for storing per-slice change detection results."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        create_command = """
        CREATE TABLE IF NOT EXISTS change_analysis (
            id SERIAL PRIMARY KEY,
            location_name TEXT NOT NULL,
            scan_time_1 TIMESTAMP NOT NULL,
            scan_time_2 TIMESTAMP NOT NULL,
            slice_id TEXT NOT NULL,
            category TEXT NOT NULL,
            object_name TEXT NOT NULL,
            change_status TEXT NOT NULL,
            visual_attributes TEXT,
            activity_status TEXT,
            slice_image_1_path TEXT,
            slice_image_2_path TEXT,
            latitude REAL,
            longitude REAL,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """
        cur.execute(create_command)
        conn.commit()
        cur.close()
        print("Table 'change_analysis' created/verified.")
    except (Exception, Error) as error:
        print(f"Error creating change_analysis table: {error}")
    finally:
        if conn is not None:
            conn.close()

def save_change_analysis(record):
    """Save a single change analysis record to the database."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        insert_command = """
        INSERT INTO change_analysis (
            location_name, scan_time_1, scan_time_2, slice_id, category,
            object_name, change_status, visual_attributes, activity_status,
            slice_image_1_path, slice_image_2_path, latitude, longitude
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        cur.execute(insert_command, (
            record['location_name'],
            record['scan_time_1'],
            record['scan_time_2'],
            record['slice_id'],
            record['category'],
            record['object_name'],
            record['change_status'],
            record.get('visual_attributes', ''),
            record.get('activity_status', ''),
            record.get('slice_image_1_path', ''),
            record.get('slice_image_2_path', ''),
            record.get('latitude'),
            record.get('longitude')
        ))
        
        record_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        return record_id
    except (Exception, Error) as error:
        print(f"Error saving change analysis record: {error}")
        return None
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    drop_old_tables()
    create_change_reports_table()
    create_change_analysis_table()
    print("Database restructured. Ready for dynamic table creation with geographic coordinates.")
