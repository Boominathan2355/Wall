"""Wall Detection System - Main inference and analysis pipeline"""
import torch
import cv2
import numpy as np
from scipy import ndimage
from import_modules import device
from wallsegmentation import UNetWallSegmenter
from config import MODEL_CONFIG, INFERENCE_CONFIG
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WallDetectionSystem:
    """Complete wall detection and analysis system"""
    
    def __init__(self, model_path=None):
        """
        Initialize the wall detection system
        
        Args:
            model_path (str): Path to trained model. If None, creates untrained model.
        """
        self.device = device
        self.model = UNetWallSegmenter(
            in_channels=MODEL_CONFIG['in_channels'],
            out_channels=MODEL_CONFIG['out_channels']
        ).to(self.device)
        self.model.eval()
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Model loaded successfully from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
        else:
            logger.warning("No model path provided or file not found. Using untrained model.")
    
    def predict_mask(self, image, return_numpy=True):
        """
        Predict wall segmentation mask for an image
        
        Args:
            image (np.ndarray): Input image (RGB, 0-255)
            return_numpy (bool): Return as numpy array if True, else tensor
            
        Returns:
            np.ndarray or torch.Tensor: Segmentation mask (same spatial size as input)
        """
        try:
            original_height, original_width = image.shape[:2]
            
            # Normalize and resize
            image_resized = cv2.resize(image, (MODEL_CONFIG['image_size'], MODEL_CONFIG['image_size']))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_normalized = (image_normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # Convert to tensor and ensure proper dtype
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            # Inference
            with torch.no_grad():
                mask_tensor = self.model(image_tensor).float()
            
            # Resize back to original dimensions
            mask_resized = torch.nn.functional.interpolate(
                mask_tensor,
                size=(original_height, original_width),
                mode='bilinear',
                align_corners=False
            )
            
            if return_numpy:
                mask = (mask_resized.squeeze().cpu().numpy() * 255).astype(np.uint8)
                return mask
            else:
                return mask_resized.squeeze()
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def calculate_area(self, mask, scale_factor=100):
        """
        Calculate wall area from binary mask
        
        Args:
            mask (np.ndarray): Binary segmentation mask
            scale_factor (float): Pixels per meter
            
        Returns:
            dict: Area statistics in pixels, square meters, and square feet
        """
        try:
            # Convert to binary
            binary_mask = (mask > 127).astype(np.uint8)
            
            # Count white pixels
            pixel_count = np.sum(binary_mask)
            
            # Calculate areas
            pixel_area = pixel_count
            real_area_m2 = (pixel_area / (scale_factor ** 2)) if scale_factor > 0 else 0
            
            # Convert square meters to square feet (1 m² = 10.764 sq ft)
            real_area_sqft = real_area_m2 * 10.764
            
            # Calculate coverage percentage
            total_pixels = mask.shape[0] * mask.shape[1]
            coverage_percentage = (pixel_count / total_pixels) * 100
            
            return {
                'pixel_area': int(pixel_area),
                'real_area_m2': float(real_area_m2),
                'real_area_sqft': float(real_area_sqft),
                'coverage_percentage': float(coverage_percentage),
                'total_pixels': int(total_pixels)
            }
        except Exception as e:
            logger.error(f"Error calculating area: {e}")
            raise
    
    def detect_individual_walls(self, mask, min_area=None):
        """
        Detect individual walls as separate connected components
        
        Args:
            mask (np.ndarray): Binary segmentation mask
            min_area (int): Minimum wall area in pixels. Uses config if None.
            
        Returns:
            list: List of wall dictionaries with properties
        """
        try:
            if min_area is None:
                min_area = INFERENCE_CONFIG['min_wall_area']
            
            # Convert to binary
            binary_mask = (mask > 127).astype(np.uint8)
            
            # Label connected components
            labeled_array, num_features = ndimage.label(binary_mask)
            
            walls = []
            for wall_id in range(1, num_features + 1):
                wall_mask = (labeled_array == wall_id).astype(np.uint8)
                area = np.sum(wall_mask)
                
                if area >= min_area:
                    # Get bounding box
                    y_coords, x_coords = np.nonzero(wall_mask)
                    if len(y_coords) > 0:
                        bbox = {
                            'x_min': int(x_coords.min()),
                            'y_min': int(y_coords.min()),
                            'x_max': int(x_coords.max()),
                            'y_max': int(y_coords.max()),
                        }
                        bbox['width'] = bbox['x_max'] - bbox['x_min']
                        bbox['height'] = bbox['y_max'] - bbox['y_min']
                        
                        walls.append({
                            'id': wall_id,
                            'area': int(area),
                            'bbox': bbox,
                            'mask': wall_mask,
                            'centroid': (
                                int(np.mean(x_coords)),
                                int(np.mean(y_coords))
                            )
                        })
            
            return walls
        except Exception as e:
            logger.error(f"Error detecting individual walls: {e}")
            raise
    
    def create_visualization(self, image, mask, walls, area_results, scale_factor=100):
        """
        Create visualization with detected walls overlaid on image
        
        Args:
            image (np.ndarray): Original image (BGR)
            mask (np.ndarray): Segmentation mask
            walls (list): List of detected walls
            area_results (dict): Area calculation results
            scale_factor (float): Pixels per meter
            
        Returns:
            np.ndarray: Visualization image (BGR)
        """
        try:
            result = image.copy()
            
            # Create colored mask overlay
            binary_mask = (mask > 127).astype(np.uint8)
            colored_mask = np.zeros_like(image)
            colored_mask[binary_mask == 1] = [0, 255, 0]  # Green for walls
            
            # Blend
            result = cv2.addWeighted(result, 0.7, colored_mask, 0.3, 0)
            
            # Draw individual walls with different colors
            colors = [
                (255, 0, 0),    # Blue
                (0, 255, 255),  # Cyan
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 0),    # Green
            ]
            
            for idx, wall in enumerate(walls):
                color = colors[idx % len(colors)]
                bbox = wall['bbox']
                
                # Draw bounding box
                cv2.rectangle(
                    result,
                    (bbox['x_min'], bbox['y_min']),
                    (bbox['x_max'], bbox['y_max']),
                    color, 2
                )
                
                # Draw wall ID
                cv2.putText(
                    result,
                    f"Wall {wall['id']}",
                    (bbox['x_min'], bbox['y_min'] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                
                # Draw centroid
                centroid = wall['centroid']
                cv2.circle(result, centroid, 5, color, -1)
            
            # Add text information
            text_lines = [
                f"Total Wall Area: {area_results['pixel_area']} pixels",
                f"Coverage: {area_results['coverage_percentage']:.2f}%",
                f"Real Area: {area_results['real_area_m2']:.2f} m² ({area_results['real_area_sqft']:.2f} sq ft)",
                f"Walls Detected: {len(walls)}"
            ]
            
            y_offset = 30
            for line in text_lines:
                cv2.putText(
                    result,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                y_offset += 25
            
            return result
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            raise
    
    def analyze_image(self, image_path, scale_factor=100, output_path=None):
        """
        Complete analysis pipeline for a single image
        
        Args:
            image_path (str): Path to input image
            scale_factor (float): Pixels per meter
            output_path (str): Path to save result image. If None, no save.
            
        Returns:
            dict: Complete analysis results
        """
        try:
            # Load image
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Predict mask
            mask = self.predict_mask(image_rgb)
            
            # Calculate area
            area_results = self.calculate_area(mask, scale_factor)
            
            # Detect individual walls
            walls = self.detect_individual_walls(mask)
            
            # Create visualization
            result_image = self.create_visualization(
                image_bgr, mask, walls, area_results, scale_factor
            )
            
            # Save if requested
            if output_path:
                cv2.imwrite(output_path, result_image)
                logger.info(f"Result saved to {output_path}")
            
            return {
                'mask': mask,
                'area_results': area_results,
                'individual_walls': walls,
                'result_image': result_image,
                'image_path': image_path
            }
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
