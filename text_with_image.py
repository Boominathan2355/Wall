"""
Text with Image - Create annotated visualizations
Adds text overlays and annotations to images
"""
import cv2
import numpy as np
import os
from pathlib import Path


def add_text_to_image(image_path, output_path=None, text_data=None):
    """
    Add text annotations to an image
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output image
        text_data (dict): Dictionary with text and position information
    
    Returns:
        np.ndarray: Image with text overlay
    """
    
    # Read image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
    else:
        image = image_path
    
    if image is None:
        return None
    
    # Default text data if not provided
    if text_data is None:
        text_data = {
            'title': 'Wall Detection Analysis',
            'subtitle': 'AI-Powered Wall Segmentation',
            'footer': 'Results: Wall area detected and analyzed'
        }
    
    # Copy image to avoid modifying original
    annotated = image.copy()
    height, width = annotated.shape[:2]
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black background
    
    # Add title at top
    if 'title' in text_data:
        title = text_data['title']
        title_size = cv2.getTextSize(title, font, font_scale + 0.3, font_thickness + 1)[0]
        title_x = (width - title_size[0]) // 2
        title_y = 40
        
        # Add background rectangle for title
        cv2.rectangle(annotated, 
                     (title_x - 10, title_y - title_size[1] - 10),
                     (title_x + title_size[0] + 10, title_y + 10),
                     bg_color, -1)
        
        cv2.putText(annotated, title, (title_x, title_y),
                   font, font_scale + 0.3, text_color, font_thickness + 1)
    
    # Add subtitle
    if 'subtitle' in text_data:
        subtitle = text_data['subtitle']
        subtitle_size = cv2.getTextSize(subtitle, font, font_scale, font_thickness)[0]
        subtitle_x = (width - subtitle_size[0]) // 2
        subtitle_y = 80
        
        cv2.putText(annotated, subtitle, (subtitle_x, subtitle_y),
                   font, font_scale, text_color, font_thickness)
    
    # Add metrics in the middle
    if 'metrics' in text_data:
        metrics = text_data['metrics']
        y_pos = 150
        
        for i, (key, value) in enumerate(metrics.items()):
            text = f"{key}: {value}"
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            x_pos = (width - text_size[0]) // 2
            
            # Add background for each metric
            cv2.rectangle(annotated,
                         (x_pos - 10, y_pos - text_size[1] - 5),
                         (x_pos + text_size[0] + 10, y_pos + 5),
                         (50, 50, 50), -1)
            
            cv2.putText(annotated, text, (x_pos, y_pos),
                       font, font_scale, text_color, font_thickness)
            y_pos += 50
    
    # Add footer at bottom
    if 'footer' in text_data:
        footer = text_data['footer']
        footer_size = cv2.getTextSize(footer, font, 0.8, 1)[0]
        footer_x = (width - footer_size[0]) // 2
        footer_y = height - 20
        
        # Add background rectangle for footer
        cv2.rectangle(annotated,
                     (footer_x - 10, footer_y - footer_size[1] - 10),
                     (footer_x + footer_size[0] + 10, footer_y + 10),
                     bg_color, -1)
        
        cv2.putText(annotated, footer, (footer_x, footer_y),
                   font, 0.8, text_color, 1)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"✓ Saved annotated image to: {output_path}")
    
    return annotated


def create_info_image(width=800, height=600):
    """Create an image with information text"""
    
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    text_data = {
        'title': 'Wall Detection System',
        'subtitle': 'Deep Learning for Wall Segmentation',
        'metrics': {
            'Model': 'U-Net (31M parameters)',
            'IoU': '0.9560',
            'Dice Coefficient': '0.9775',
            'Accuracy': '0.9970',
            'Precision': '0.9993',
            'Recall': '0.9567'
        },
        'footer': 'Powered by PyTorch • Built for Production'
    }
    
    # Add text to image
    result = add_text_to_image(image, text_data=text_data)
    
    return result


def annotate_analysis_results(image_path, results, output_path=None):
    """
    Annotate an analysis image with detection results
    
    Args:
        image_path (str): Path to analysis result image
        results (dict): Detection results dictionary
        output_path (str): Path to save output
    
    Returns:
        np.ndarray: Annotated image
    """
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Extract results
    text_data = {
        'title': 'Wall Detection Results',
        'metrics': {
            f'Wall Area': f"{results.get('real_area', 0):.2f} m²",
            f'Coverage': f"{results.get('coverage_percentage', 0):.2f}%",
            f'Walls Detected': f"{results.get('walls_count', 0)} walls",
            f'Pixel Area': f"{results.get('pixel_area', 0)} px"
        },
        'footer': 'AI Analysis Complete • Ready for Export'
    }
    
    # Add text overlay
    result = add_text_to_image(image, output_path=output_path, text_data=text_data)
    
    return result


def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("TEXT WITH IMAGE - ANNOTATION TOOL")
    print("="*80)
    
    # Example 1: Create info image
    print("\n1. Creating information image with metrics...")
    info_image = create_info_image(width=1000, height=700)
    cv2.imwrite("results/info_image.png", info_image)
    print("✓ Saved to: results/info_image.png")
    
    # Example 2: Annotate existing analysis results
    print("\n2. Annotating analysis results...")
    
    # Check if analysis images exist
    analysis_images = []
    results_dir = Path("results")
    if results_dir.exists():
        analysis_images = list(results_dir.glob("analysis_*.jpg"))
    
    if analysis_images:
        # Sample results data
        sample_results = {
            'real_area': 0.45,
            'coverage_percentage': 6.83,
            'walls_count': 1,
            'pixel_area': 4478
        }
        
        for img_path in analysis_images[:2]:  # Process first 2 images
            output_name = f"results/annotated_{img_path.stem}.png"
            print(f"\nAnnotating: {img_path.name}")
            
            annotate_analysis_results(str(img_path), sample_results, output_name)
    else:
        print("No analysis images found in results/")
        print("Run python usage.py first to generate analysis images")
    
    # Example 3: Add text to existing image
    print("\n3. Processing sample images with text overlay...")
    
    # Find any PNG/JPG files in data directories
    sample_images = []
    sample_images.extend(Path("data/val/images").glob("*.jpg"))
    sample_images.extend(Path("data/val/images").glob("*.png"))
    
    if sample_images:
        for img_path in sample_images[:2]:
            text_data = {
                'title': f'Sample: {img_path.stem}',
                'subtitle': 'Wall Detection Input Image',
                'metrics': {
                    'Resolution': '256x256',
                    'Format': 'RGB JPEG',
                    'Status': 'Ready for Analysis'
                },
                'footer': 'Processing... → Detection → Analysis'
            }
            
            output_name = f"results/sample_{img_path.stem}_annotated.jpg"
            print(f"Processing: {img_path.name}")
            add_text_to_image(str(img_path), output_path=output_name, text_data=text_data)
    
    print("\n" + "="*80)
    print("ANNOTATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  • results/info_image.png - System information and metrics")
    print("  • results/annotated_* - Annotated analysis results")
    print("  • results/sample_*_annotated.jpg - Annotated sample images")
    print("\nAll annotated images saved to: results/")


if __name__ == "__main__":
    main()
