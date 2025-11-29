from wall_detection_system import WallDetectionSystem
import cv2
import matplotlib.pyplot as plt
import os
import glob


def main():
    """Analyze images from our sample dataset"""
    
    print("\n" + "="*80)
    print("WALL DETECTION SYSTEM - INFERENCE EXAMPLES")
    print("="*80)
    
    # Initialize the system with our trained model
    print("\nInitializing model...")
    wall_system = WallDetectionSystem(model_path="models/quick_trained_model.pth")
    
    # Example 1: Analyze test images from our dataset
    print("\n" + "-"*80)
    print("Example 1: Analyzing test images from sample dataset")
    print("-"*80)
    
    # Get test images
    test_images = glob.glob("data/test/images/*.jpg") + glob.glob("data/test/images/*.png")
    
    if not test_images:
        # Use validation images if no test images
        test_images = glob.glob("data/val/images/*.jpg") + glob.glob("data/val/images/*.png")
        print("No test images found, using validation images instead")
    
    if test_images:
        for i, image_path in enumerate(test_images[:3]):  # Analyze first 3 images
            print(f"\nAnalyzing image {i+1}: {os.path.basename(image_path)}")
            
            try:
                # Create output directory
                os.makedirs("results", exist_ok=True)
                
                # Analyze image
                output_path = f"results/analysis_{i:03d}.jpg"
                results = wall_system.analyze_image(
                    image_path=image_path,
                    scale_factor=100,  # 100 pixels = 1 meter
                    output_path=output_path
                )
                
                # Display results
                print(f"  ✓ Analysis complete")
                print(f"  Total wall area: {results['area_results']['pixel_area']} pixels")
                print(f"  Coverage: {results['area_results']['coverage_percentage']:.2f}%")
                print(f"  Real area: {results['area_results']['real_area']:.2f} m²")
                print(f"  Individual walls detected: {len(results['individual_walls'])}")
                print(f"  Result saved to: {output_path}")
                
                for wall in results['individual_walls'][:3]:  # Show first 3 walls
                    print(f"\n    Wall {wall['id']}:")
                    print(f"      - Area: {wall['area']} pixels")
                    
            except Exception as e:
                print(f"  ✗ Error analyzing image: {e}")
    else:
        print("\n✗ No images found in data/test/images or data/val/images")
        print("Please ensure the dataset is properly set up")
    
    # Example 2: Get model predictions
    print("\n" + "-"*80)
    print("Example 2: Model Predictions on First Image")
    print("-"*80)
    
    if test_images:
        try:
            import torch
            import numpy as np
            
            image_path = test_images[0]
            print(f"\nProcessing: {os.path.basename(image_path)}")
            
            # Get prediction
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = wall_system.predict_mask(image_rgb)
            
            print(f"  ✓ Prediction shape: {mask.shape}")
            print(f"  Mask value range: [{mask.min():.4f}, {mask.max():.4f}]")
            print(f"  Detected wall pixels: {(mask > 0.5).sum()} / {mask.size}")
            print(f"  Wall coverage: {(mask > 0.5).sum() / mask.size * 100:.2f}%")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print("\nResults saved to: results/")
    print("\nNext steps:")
    print("  • Review the analysis results in the results/ folder")
    print("  • Train on a larger dataset for better accuracy")
    print("  • Adjust scale_factor based on your actual camera calibration")


def batch_processing():
    """Process multiple images in batch"""
    wall_system = WallDetectionSystem(model_path="models/quick_trained_model.pth")
    
    image_folder = "data/val/images/"
    output_folder = "results/"
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(image_folder):
        print(f"Error: {image_folder} does not exist")
        return
    
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    for img_file in image_files:
        print(f"Processing {img_file}...")
        
        input_path = os.path.join(image_folder, img_file)
        output_path = os.path.join(output_folder, f"analyzed_{img_file}")
        
        try:
            results = wall_system.analyze_image(
                image_path=input_path,
                scale_factor=100,  # Adjust based on your images
                output_path=output_path
            )
            
            print(f"  ✓ Walls detected: {len(results['individual_walls'])}")
            print(f"  ✓ Total area: {results['area_results']['pixel_area']} pixels")
            print(f"  ✓ Coverage: {results['area_results']['coverage_percentage']:.2f}%")
        
        except Exception as e:
            print(f"  ✗ Error processing {img_file}: {e}")


if __name__ == "__main__":
    # Run main analysis
    main()
    # Uncomment to run batch processing
    # batch_processing()