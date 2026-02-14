from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from PIL import Image, ImageDraw
import numpy as np
import rasterio
from rasterio.warp import transform as rasterio_transform
from rasterio.crs import CRS
import cv2
import random

class objectDetector:
    def __init__(self, model_path, model_type='yolov8', confidence_threshold=0.3, device="cuda:0"):
        """
        Initialize the object detection model
        """
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        print(f"Initializing model: {model_path}")
        
        # Define colors for different object classes
        self.class_colors = {}
        self.color_palette = [
            (0, 255, 0),      # Green
            (255, 0, 0),      # Red
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
        ]

    def get_class_color(self, class_name):
        """Get consistent color for each class."""
        if class_name not in self.class_colors:
            color_idx = len(self.class_colors) % len(self.color_palette)
            self.class_colors[class_name] = self.color_palette[color_idx]
        return self.class_colors[class_name]

    def predict(self, image_path, slice_height=240, slice_width=240, overlap_height_ratio=0.2, overlap_width_ratio=0.2):
        """
        Performs sliced object detection on the input image.
        """
        print(f"Starting sliced prediction for: {image_path}")
        result = get_sliced_prediction(
            image_path,
            self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        return result

    def visualize_and_save(self, result, original_image_path, output_path):
        """
        Creates OBB visualization by duplicating the original GeoTIFF and drawing on it.
        This preserves all geographic metadata (coordinates, projection, etc.)
        """
        print(f"Creating OBB visualization and saving to: {output_path}")
        
        try:
            import shutil
            
            # Step 1: Copy the original GeoTIFF to preserve metadata
            shutil.copy2(original_image_path, output_path)
            print(f"  → Duplicated original GeoTIFF")
            
            # Step 2: Open with rasterio and OpenCV for OBB drawing
            with rasterio.open(output_path, 'r+') as dst:
                # Read the image bands
                img_array = dst.read()
                
                # Convert to OpenCV format for OBB drawing
                if img_array.shape[0] == 1:
                    # Grayscale to BGR
                    img_cv = cv2.cvtColor(img_array[0], cv2.COLOR_GRAY2BGR)
                elif img_array.shape[0] >= 3:
                    # RGB to BGR
                    img_cv = cv2.cvtColor(np.transpose(img_array[:3], (1, 2, 0)), cv2.COLOR_RGB2BGR)
                else:
                    print(f"  → Unsupported band count: {img_array.shape[0]}")
                    return
                
                # Step 3: Draw OBB (Oriented Bounding Boxes) with color coding
                for obj in result.object_prediction_list:
                    category = obj.category.name
                    score = obj.score.value
                    
                    # Get color for this class
                    color = self.get_class_color(category)
                    
                    # Get bbox - check if it's OBB or HBB
                    if hasattr(obj.bbox, 'to_xywhr'):
                        # OBB format
                        bbox_data = obj.bbox.to_xywhr()
                        x_center, y_center, width, height, rotation = bbox_data
                        
                        # Create rotated rectangle
                        box = cv2.boxPoints(((x_center, y_center), (width, height), rotation))
                        box = np.int0(box)
                        
                        # Draw OBB
                        cv2.drawContours(img_cv, [box], 0, color, 2)
                    else:
                        # Fallback to regular bbox if OBB not available
                        bbox = obj.bbox.to_xyxy()
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # Draw label
                    label = f"{category} {score:.2f}"
                    bbox_xyxy = obj.bbox.to_xyxy()
                    x_min, y_min = int(bbox_xyxy[0]), int(bbox_xyxy[1])
                    
                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_cv, (x_min, y_min - text_height - 5), (x_min + text_width, y_min), color, -1)
                    cv2.putText(img_cv, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Step 4: Convert back and write to GeoTIFF
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_np = np.transpose(img_rgb, (2, 0, 1))
                
                # Write back to the file (preserving all metadata)
                dst.write(img_np[:dst.count])
                
            print(f"  → Saved with preserved GeoTIFF metadata and OBB visualization")
            
        except Exception as e:
            print(f"Error in OBB visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def crop_object(self, original_image_path, bbox, output_path):
        """
        Crops a single object from the original image using bounding box coordinates.
        """
        try:
            image = Image.open(original_image_path)
            cropped = image.crop(bbox)
            cropped.save(output_path)
        except Exception as e:
            print(f"Error cropping object: {e}")
    
    def get_geotransform_and_crs(self, image_path):
        """
        Extract the geotransform and CRS from a GeoTIFF file.
        
        Returns:
            Tuple of (transform, crs) or (None, None) if not a GeoTIFF
        """
        try:
            with rasterio.open(image_path) as src:
                return src.transform, src.crs
        except Exception as e:
            print(f"Warning: Could not read geotransform: {e}")
            return None, None
    
    def pixel_to_latlon(self, pixel_x, pixel_y, transform, src_crs):
        """
        Convert pixel coordinates to WGS84 latitude/longitude.
        
        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            transform: Rasterio transform object
            src_crs: Source coordinate reference system
            
        Returns:
            Tuple of (latitude, longitude) in WGS84
        """
        if transform is None or src_crs is None:
            return None, None
        
        try:
            # Convert pixel to source CRS coordinates
            x, y = rasterio.transform.xy(transform, pixel_y, pixel_x)
            
            # Transform to WGS84 (EPSG:4326) lat/lon
            lon, lat = rasterio_transform(src_crs, CRS.from_epsg(4326), [x], [y])
            
            return lat[0], lon[0]
        except Exception as e:
            print(f"Error converting coordinates: {e}")
            return None, None

