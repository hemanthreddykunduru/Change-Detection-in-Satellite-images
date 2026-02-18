"""
Satellite Image Change Detection System
========================================
Pairs consecutive satellite images, slices them into 240x240 tiles,
queries the DB for detected objects per slice, sends slice pairs to
Ollama gemma3:4b for visual comparison, and stores results in
PostgreSQL + .txt reports.
"""

import os
import glob
import json
import base64
import math
import requests
from datetime import datetime
from PIL import Image
import rasterio
from rasterio.warp import transform as rasterio_transform
from rasterio.crs import CRS
from dotenv import load_dotenv
from database import (
    create_change_analysis_table,
    save_change_analysis,
    query_objects_in_bounds,
    DB_CONFIG
)

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
CROPPED_FOLDER = os.getenv("INPUT_FOLDER", r"H:\change detection\cropped")
BASE_DIR = os.getenv("DETECTED_FOLDER", r"H:\change detection")
SLICE_SIZE = 240  # pixels
LOCATION_NAME = "Mundra Port"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"

SLICE_OUTPUT_DIR = os.path.join(BASE_DIR, "slice_images")
REPORT_OUTPUT_DIR = os.path.join(BASE_DIR, "change_reports")


# ============================================================
# UTILITY: Parse timestamp from filename like 20260114T055059.tif
# ============================================================
def parse_timestamp_from_filename(filename):
    """Parse datetime from filename format: 20260114T055059.tif"""
    basename = os.path.splitext(os.path.basename(filename))[0]
    try:
        return datetime.strptime(basename, "%Y%m%dT%H%M%S")
    except ValueError:
        print(f"Warning: Could not parse timestamp from {filename}")
        return None


# ============================================================
# UTILITY: Get table name from image filename (matches database.py)
# ============================================================
def get_table_name(filename):
    """Convert filename to table name (same logic as database.py)."""
    return os.path.basename(filename).replace('.tif', '')


# ============================================================
# UTILITY: Compute WGS84 coordinates for a pixel location
# ============================================================
def pixel_to_wgs84(col, row, src_transform, src_crs):
    """Convert pixel (col, row) to WGS84 (lon, lat)."""
    x, y = src_transform * (col, row)
    lon, lat = rasterio_transform(src_crs, CRS.from_epsg(4326), [x], [y])
    return lon[0], lat[0]


# ============================================================
# SLICE IMAGE INTO 240x240 TILES
# ============================================================
def slice_image(image_path, output_dir, slice_size=240):
    """
    Slice a GeoTIFF into non-overlapping tiles of slice_size x slice_size.
    Returns list of dicts with slice metadata including geo-coordinates.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Open with rasterio for coordinate info
    with rasterio.open(image_path) as src:
        src_transform = src.transform
        src_crs = src.crs
        img_width = src.width
        img_height = src.height

    # Open with PIL for slicing
    img = Image.open(image_path)

    num_rows = math.ceil(img_height / slice_size)
    num_cols = math.ceil(img_width / slice_size)

    slices = []

    for r in range(num_rows):
        for c in range(num_cols):
            x_start = c * slice_size
            y_start = r * slice_size
            x_end = min(x_start + slice_size, img_width)
            y_end = min(y_start + slice_size, img_height)

            # Crop the slice
            slice_img = img.crop((x_start, y_start, x_end, y_end))

            # Pad to exact slice_size if at the edge
            if slice_img.size != (slice_size, slice_size):
                padded = Image.new(slice_img.mode, (slice_size, slice_size), 0)
                padded.paste(slice_img, (0, 0))
                slice_img = padded

            # Save slice
            slice_name = f"slice_r{r}_c{c}.png"
            slice_path = os.path.join(output_dir, slice_name)
            slice_img.save(slice_path)

            # Compute WGS84 bounds for this slice
            tl_lon, tl_lat = pixel_to_wgs84(x_start, y_start, src_transform, src_crs)
            br_lon, br_lat = pixel_to_wgs84(x_end, y_end, src_transform, src_crs)

            # Center coordinate
            center_lon = (tl_lon + br_lon) / 2.0
            center_lat = (tl_lat + br_lat) / 2.0

            # Min/max for DB query (lat might be inverted)
            min_lat = min(tl_lat, br_lat)
            max_lat = max(tl_lat, br_lat)
            min_lon = min(tl_lon, br_lon)
            max_lon = max(tl_lon, br_lon)

            slices.append({
                'slice_id': f"R{r}C{c}",
                'slice_name': slice_name,
                'slice_path': slice_path,
                'row': r,
                'col': c,
                'pixel_bounds': (x_start, y_start, x_end, y_end),
                'center_lat': center_lat,
                'center_lon': center_lon,
                'min_lat': min_lat,
                'max_lat': max_lat,
                'min_lon': min_lon,
                'max_lon': max_lon
            })

    print(f"  Sliced into {len(slices)} tiles ({num_rows} rows x {num_cols} cols)")
    return slices


# ============================================================
# ENCODE IMAGE TO BASE64 FOR OLLAMA
# ============================================================
def image_to_base64(image_path):
    """Read image file and return base64 encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================
# QUERY OLLAMA WITH SLICE PAIR + DB INFO
# ============================================================
def analyze_slice_pair(before_path, after_path, before_objects, after_objects, slice_id):
    """
    Send before/after slice images + DB object info to Ollama gemma3:4b.
    Returns parsed JSON response.
    """
    # Build object info strings
    before_obj_str = "No objects detected in database for this area" if not before_objects else json.dumps(
        [{"id": o["id"], "name": o["object_name"], "accuracy": o["accuracy_percentage"]}
         for o in before_objects], indent=2
    )
    after_obj_str = "No objects detected in database for this area" if not after_objects else json.dumps(
        [{"id": o["id"], "name": o["object_name"], "accuracy": o["accuracy_percentage"]}
         for o in after_objects], indent=2
    )

    prompt = f"""You are a military-grade satellite imagery analyst performing change detection on slice {slice_id}.

You are given two satellite images:
- Image 1 = BEFORE (earlier scan)
- Image 2 = AFTER (later scan)

Known database objects in BEFORE scan area: {before_obj_str}
Known database objects in AFTER scan area: {after_obj_str}

YOUR TASK: Look at BOTH images very carefully. Describe EVERYTHING you see. Then compare them.

MANDATORY RULES:
- You MUST describe what you see in the images. NEVER say NONE as object_name.
- You MUST name visible features: water, land, buildings, ships, roads, docks, cranes, tanks, vegetation, sand, concrete, etc.
- You MUST compare the two images and describe ANY difference no matter how small.
- Even if both images look similar, describe what is visible and say the objects are STATIONARY.
- The object_name field MUST contain the most prominent visible feature (e.g., "Water body", "Dock area", "Ships", "Buildings", "Open land", "Road", "Storage tanks").
- The visual_attributes field MUST describe what you actually see in both images in detail.
- The activity_status field MUST describe what changed or say "No visible change detected between scans".

DETECTION FOCUS:
1. Infrastructure: construction, demolition, excavation, land clearing, new buildings, road extension
2. Industrial: storage tanks, warehouse expansion, equipment placement
3. Maritime: ship movement, new vessels, vessels departed, docking changes
4. Ground: soil disturbance, color changes, new paths, cleared areas

CHANGE STATUS VALUES:
- STATIONARY = object exists in both, no change
- MOVED = object shifted position
- EXPANDED = object/area grew larger
- REDUCED = object/area got smaller
- NEW = appears only in AFTER image
- REMOVED = exists only in BEFORE image
- NO_CHANGE = entire scene is identical
- UNCERTAIN = cannot determine clearly

DO NOT add explanation outside JSON. DO NOT add markdown. DO NOT add comments.

Respond with ONLY this JSON:
{{"category": "INFRASTRUCTURE|MARITIME|INDUSTRIAL|SECURITY|NO_CHANGE", "object_name": "name of most prominent visible feature", "change_status": "STATIONARY|MOVED|EXPANDED|REDUCED|NEW|REMOVED|NO_CHANGE|UNCERTAIN", "visual_attributes": "describe what you see in both images", "activity_status": "describe what changed between the two images"}}"""

    # Encode images
    before_b64 = image_to_base64(before_path)
    after_b64 = image_to_base64(after_path)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [before_b64, after_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 500
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        raw_text = result.get("response", "").strip()

        # Clean any markdown code fences if model adds them despite instructions
        cleaned = raw_text
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        # Try to extract JSON from the response
        try:
            parsed = json.loads(cleaned)
            return parsed
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(cleaned[start:end])
                    return parsed
                except json.JSONDecodeError:
                    pass

            print(f"    Warning: Could not parse JSON from model response for {slice_id}")
            print(f"    Raw response: {raw_text[:200]}")
            return {
                "category": "UNCERTAIN",
                "object_name": "Unknown",
                "change_status": "UNCERTAIN",
                "visual_attributes": f"Model response not parseable: {raw_text[:100]}",
                "activity_status": "UNCERTAIN"
            }

    except requests.exceptions.RequestException as e:
        print(f"    Error calling Ollama for {slice_id}: {e}")
        return {
            "category": "UNCERTAIN",
            "object_name": "Unknown",
            "change_status": "UNCERTAIN",
            "visual_attributes": f"Ollama error: {str(e)[:100]}",
            "activity_status": "Error"
        }


# ============================================================
# PROCESS ONE IMAGE PAIR
# ============================================================
def process_image_pair(before_path, after_path):
    """
    Process a single pair of consecutive satellite images.
    Slices both, queries DB, analyzes with LLM, saves results.
    """
    before_filename = os.path.basename(before_path)
    after_filename = os.path.basename(after_path)
    before_ts = parse_timestamp_from_filename(before_filename)
    after_ts = parse_timestamp_from_filename(after_filename)

    if before_ts is None or after_ts is None:
        print(f"Skipping pair: could not parse timestamps")
        return

    before_table = get_table_name(before_filename)
    after_table = get_table_name(after_filename)

    # Create folder structure for slice images
    pair_folder_name = f"{before_ts.strftime('%Y%m%dT%H%M%S')}_to_{after_ts.strftime('%Y%m%dT%H%M%S')}"
    before_slice_dir = os.path.join(SLICE_OUTPUT_DIR, pair_folder_name, "before")
    after_slice_dir = os.path.join(SLICE_OUTPUT_DIR, pair_folder_name, "after")

    print(f"\n{'='*70}")
    print(f"PAIR: {before_filename} → {after_filename}")
    print(f"{'='*70}")

    # Step 1: Slice both images
    print(f"\n[1/4] Slicing BEFORE image: {before_filename}")
    before_slices = slice_image(before_path, before_slice_dir, SLICE_SIZE)

    print(f"[2/4] Slicing AFTER image: {after_filename}")
    after_slices = slice_image(after_path, after_slice_dir, SLICE_SIZE)

    # Build report content
    report_lines = []
    report_lines.append(f"CHANGE DETECTION REPORT")
    report_lines.append(f"=" * 70)
    report_lines.append(f"Location: {LOCATION_NAME}")
    report_lines.append(f"Before: {before_filename} ({before_ts})")
    report_lines.append(f"After:  {after_filename} ({after_ts})")
    report_lines.append(f"Analysis Time: {datetime.now()}")
    report_lines.append(f"Slice Size: {SLICE_SIZE}x{SLICE_SIZE} pixels")
    report_lines.append(f"Total Slices: {len(before_slices)}")
    report_lines.append(f"=" * 70)
    report_lines.append("")

    total_db_records = 0

    # Step 2: Process each slice pair
    print(f"\n[3/4] Analyzing {len(before_slices)} slice pairs with {OLLAMA_MODEL}...")

    for i, (b_slice, a_slice) in enumerate(zip(before_slices, after_slices)):
        slice_id = b_slice['slice_id']
        print(f"\n  --- Slice {slice_id} ({i+1}/{len(before_slices)}) ---")

        # Query DB for objects in this slice's geographic bounds
        before_objects = query_objects_in_bounds(
            before_table,
            b_slice['min_lat'], b_slice['max_lat'],
            b_slice['min_lon'], b_slice['max_lon']
        )
        after_objects = query_objects_in_bounds(
            after_table,
            a_slice['min_lat'], a_slice['max_lat'],
            a_slice['min_lon'], a_slice['max_lon']
        )

        print(f"    DB objects - Before: {len(before_objects)}, After: {len(after_objects)}")

        # Analyze with Ollama
        analysis = analyze_slice_pair(
            b_slice['slice_path'],
            a_slice['slice_path'],
            before_objects,
            after_objects,
            slice_id
        )

        print(f"    Result: {analysis.get('category', 'N/A')} | "
              f"{analysis.get('object_name', 'N/A')} | "
              f"{analysis.get('change_status', 'N/A')}")

        # Relative paths for DB storage
        rel_before_path = os.path.join("slice_images", pair_folder_name, "before", b_slice['slice_name'])
        rel_after_path = os.path.join("slice_images", pair_folder_name, "after", a_slice['slice_name'])

        # Save to database
        record = {
            'location_name': LOCATION_NAME,
            'scan_time_1': before_ts,
            'scan_time_2': after_ts,
            'slice_id': slice_id,
            'category': analysis.get('category', 'UNCERTAIN'),
            'object_name': analysis.get('object_name', 'Unknown'),
            'change_status': analysis.get('change_status', 'UNCERTAIN'),
            'visual_attributes': analysis.get('visual_attributes', ''),
            'activity_status': analysis.get('activity_status', ''),
            'slice_image_1_path': rel_before_path,
            'slice_image_2_path': rel_after_path,
            'latitude': b_slice['center_lat'],
            'longitude': b_slice['center_lon']
        }

        record_id = save_change_analysis(record)
        if record_id:
            total_db_records += 1
            print(f"    Saved to DB: id={record_id}")

        # Add to report
        # --- BEGIN SLICE REPORT SECTION ---
        # Comment out the lines below (up to END SLICE REPORT SECTION) to remove
        # individual slice details from the report for debugging purposes.
        report_lines.append(f"SLICE: {slice_id}")
        report_lines.append(f"-" * 40)
        report_lines.append(f"  Before Image: {rel_before_path}")
        report_lines.append(f"  After Image:  {rel_after_path}")
        report_lines.append(f"  Center Lat: {b_slice['center_lat']:.6f}")
        report_lines.append(f"  Center Lon: {b_slice['center_lon']:.6f}")
        report_lines.append(f"  Bounds: lat[{b_slice['min_lat']:.6f}, {b_slice['max_lat']:.6f}] "
                           f"lon[{b_slice['min_lon']:.6f}, {b_slice['max_lon']:.6f}]")
        report_lines.append(f"  DB Objects Before: {len(before_objects)}")
        if before_objects:
            for obj in before_objects:
                report_lines.append(f"    - {obj['id']}: {obj['object_name']} "
                                   f"(acc: {obj['accuracy_percentage']:.1f}%)")
        report_lines.append(f"  DB Objects After: {len(after_objects)}")
        if after_objects:
            for obj in after_objects:
                report_lines.append(f"    - {obj['id']}: {obj['object_name']} "
                                   f"(acc: {obj['accuracy_percentage']:.1f}%)")
        report_lines.append(f"  Category: {analysis.get('category', 'N/A')}")
        report_lines.append(f"  Object: {analysis.get('object_name', 'N/A')}")
        report_lines.append(f"  Change Status: {analysis.get('change_status', 'N/A')}")
        report_lines.append(f"  Visual Attributes: {analysis.get('visual_attributes', 'N/A')}")
        report_lines.append(f"  Activity Status: {analysis.get('activity_status', 'N/A')}")
        report_lines.append("")
        # --- END SLICE REPORT SECTION ---

    # Step 3: Save report
    print(f"\n[4/4] Saving report...")
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    report_filename = f"{before_ts.strftime('%Y%m%d_%H%M%S')}_to_{after_ts.strftime('%Y%m%d_%H%M%S')}_CHANGE_REPORT.txt"
    report_path = os.path.join(REPORT_OUTPUT_DIR, report_filename)

    # Add summary at the end
    report_lines.append(f"{'='*70}")
    report_lines.append(f"SUMMARY")
    report_lines.append(f"{'='*70}")
    report_lines.append(f"Total slices analyzed: {len(before_slices)}")
    report_lines.append(f"Total DB records created: {total_db_records}")
    report_lines.append(f"Report generated at: {datetime.now()}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"  Report saved: {report_path}")
    print(f"  DB records inserted: {total_db_records}")

    return report_path


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'#'*70}")
    print(f"# SATELLITE IMAGE CHANGE DETECTION SYSTEM")
    print(f"# Location: {LOCATION_NAME}")
    print(f"# Started: {datetime.now()}")
    print(f"{'#'*70}")

    # Ensure output directories exist
    os.makedirs(SLICE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    # Create DB table
    create_change_analysis_table()

    # Check Ollama connectivity before starting
    print(f"\nChecking Ollama at {OLLAMA_URL}...")
    try:
        test_resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        test_resp.raise_for_status()
        models = [m["name"] for m in test_resp.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"ERROR: Model '{OLLAMA_MODEL}' not found in Ollama.")
            print(f"  Available models: {models}")
            print(f"  Run: ollama pull {OLLAMA_MODEL}")
            return
        print(f"  Ollama OK. Model '{OLLAMA_MODEL}' is available.")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to Ollama at localhost:11434")
        print(f"  Start Ollama first: ollama serve")
        return
    except Exception as e:
        print(f"ERROR: Ollama check failed: {e}")
        return

    # Get all TIF images sorted chronologically
    tif_files = sorted(glob.glob(os.path.join(CROPPED_FOLDER, "*.tif")))

    if len(tif_files) < 2:
        print(f"Need at least 2 images. Found {len(tif_files)} in {CROPPED_FOLDER}")
        return

    print(f"\nFound {len(tif_files)} images:")
    for f in tif_files:
        ts = parse_timestamp_from_filename(f)
        print(f"  {os.path.basename(f)} → {ts}")

    # Create consecutive pairs: (1,2), (2,3), (3,4), ...
    pairs = [(tif_files[i], tif_files[i + 1]) for i in range(len(tif_files) - 1)]
    print(f"\nWill process {len(pairs)} consecutive pairs:")
    for before, after in pairs:
        print(f"  {os.path.basename(before)} → {os.path.basename(after)}")

    # Process each pair
    reports = []
    for pair_idx, (before_path, after_path) in enumerate(pairs):
        print(f"\n\n{'*'*70}")
        print(f"* PAIR {pair_idx + 1}/{len(pairs)}")
        print(f"{'*'*70}")

        report_path = process_image_pair(before_path, after_path)
        if report_path:
            reports.append(report_path)

    # Final summary
    print(f"\n\n{'#'*70}")
    print(f"# PIPELINE COMPLETE")
    print(f"# Pairs processed: {len(pairs)}")
    print(f"# Reports generated: {len(reports)}")
    print(f"# Slice images: {SLICE_OUTPUT_DIR}")
    print(f"# Reports: {REPORT_OUTPUT_DIR}")
    print(f"# Database: change_analysis table in {os.getenv('DB_NAME', 'detection')}")
    print(f"# Finished: {datetime.now()}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
