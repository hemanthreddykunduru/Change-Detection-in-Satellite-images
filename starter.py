import os
import glob
from detection import objectDetector
from database import drop_old_tables, create_table_for_image, save_object_to_table
from collections import Counter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    # --- Configuration from .env file ---
    input_folder = os.getenv("INPUT_FOLDER")
    detected_folder = os.getenv("DETECTED_FOLDER")
    model_path = os.getenv("MODEL_PATH")
    model_type = os.getenv("MODEL_TYPE", "yolov8")
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
    device = os.getenv("DEVICE", "cuda:0")
    slice_height = int(os.getenv("SLICE_HEIGHT", "240"))
    slice_width = int(os.getenv("SLICE_WIDTH", "240"))
    overlap_height_ratio = float(os.getenv("OVERLAP_HEIGHT_RATIO", "0.2"))
    overlap_width_ratio = float(os.getenv("OVERLAP_WIDTH_RATIO", "0.2"))
    
    # 0. Drop old database structure (one-time cleanup)
    drop_old_tables()

    # Create output directories
    output_folder = os.path.join(detected_folder, "detected")
    objects_folder = os.path.join(detected_folder, "objects")
    
    for folder in [output_folder, objects_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

    # --- Initialize Detector ---
    detector = objectDetector(
        model_path=model_path,
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        device=device
    )

    # --- Batch Process ---
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    
    if not tif_files:
        print(f"No .tif files found in {input_folder}")
        return

    print(f"Found {len(tif_files)} files to process.\n")

    for image_path in tif_files:
        filename = os.path.basename(image_path)
        print(f"{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        # Create output path for visualization
        output_tif_path = os.path.join(output_folder, filename)
        
        try:
            # 1. Run prediction
            result = detector.predict(
                image_path=image_path,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio
            )
            
            # 2. Save full visualization
            detector.visualize_and_save(
                result=result,
                original_image_path=image_path,
                output_path=output_tif_path
            )
            
            # 3. Create dynamic table for this image
            table_name = create_table_for_image(filename)
            if not table_name:
                print(f"Failed to create table for {filename}, skipping database save.")
                continue
            
            # 4. Process each detected object
            object_predictions = result.object_prediction_list
            print(f"Found {len(object_predictions)} objects.")
            
            # Get geotransform and CRS for coordinate conversion
            transform, src_crs = detector.get_geotransform_and_crs(image_path)
            if transform is None or src_crs is None:
                print(f"Warning: No geotransform/CRS found for {filename}. Skipping geographic coordinate conversion.")
                continue
            
            # Count objects by category for unique ID generation
            category_counters = Counter()
            
            for pred in object_predictions:
                # Get object details
                category = pred.category.name
                bbox = pred.bbox.to_xyxy()  # (x_min, y_min, x_max, y_max) in PIXELS
                accuracy = float(pred.score.value) * 100
                
                # Generate unique ID (e.g., ship_1, ship_2, etc.)
                category_counters[category] += 1
                object_id = f"{category}_{category_counters[category]}"
                
                # Create cropped object filename and path
                cropped_filename = f"{filename.replace('.tif', '')}_{object_id}.png"
                cropped_path = os.path.join(objects_folder, cropped_filename)
                
                # 5. Crop and save the object (using pixel coordinates)
                detector.crop_object(
                    original_image_path=image_path,
                    bbox=bbox,
                    output_path=cropped_path
                )
                
                # 6. Convert pixel coordinates to WGS84 lat/lon
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = bbox
                
                # Convert corners to lat/lon (WGS84)
                top_left_lat, top_left_lon = detector.pixel_to_latlon(top_left_x, top_left_y, transform, src_crs)
                bottom_right_lat, bottom_right_lon = detector.pixel_to_latlon(bottom_right_x, bottom_right_y, transform, src_crs)
                
                # Calculate centroid in geographic coordinates
                centroid_lat = (top_left_lat + bottom_right_lat) / 2.0
                centroid_lon = (top_left_lon + bottom_right_lon) / 2.0
                
                # Package geographic coordinates
                geo_coords = {
                    'top_left_lat': top_left_lat,
                    'top_left_lon': top_left_lon,
                    'bottom_right_lat': bottom_right_lat,
                    'bottom_right_lon': bottom_right_lon,
                    'centroid_lat': centroid_lat,
                    'centroid_lon': centroid_lon
                }
                
                # 7. Save to database with geographic coordinates
                save_object_to_table(
                    table_name=table_name,
                    object_id=object_id,
                    object_name=category,
                    geo_coords=geo_coords,
                    cropped_path=cropped_path,
                    accuracy=accuracy
                )
                
                print(f"  → {object_id}: {category} @ {accuracy:.2f}% | Lat: {centroid_lat:.4f}°, Lon: {centroid_lon:.4f}° | Saved: {cropped_filename}")
            
            print(f"✓ Completed {filename}: {len(object_predictions)} objects processed.\n")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}\n")

    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  - Visualizations: {output_folder}")
    print(f"  - Cropped objects: {objects_folder}")
    print(f"  - Database: PostgreSQL (one table per image)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
