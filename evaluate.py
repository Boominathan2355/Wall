"""Example evaluation script for wall segmentation model"""
import torch
from torch.utils.data import DataLoader
import os
import logging
from tqdm import tqdm
import numpy as np

from import_modules import device
from wallsegmentation import UNetWallSegmenter
from dataset import WallDataset
from losses_and_metrics import SegmentationMetrics
from config import MODEL_CONFIG, VALIDATION_CONFIG

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate wall segmentation model"""
    
    def __init__(self, model_path):
        """
        Initialize evaluator
        
        Args:
            model_path (str): Path to trained model
        """
        self.device = device
        self.model = UNetWallSegmenter(
            in_channels=MODEL_CONFIG['in_channels'],
            out_channels=MODEL_CONFIG['out_channels']
        ).to(self.device)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
    
    def evaluate(self, test_image_dir, test_mask_dir):
        """
        Evaluate model on test set
        
        Args:
            test_image_dir (str): Path to test images
            test_mask_dir (str): Path to test masks
            
        Returns:
            dict: Evaluation metrics
        """
        # Create dataset and dataloader
        test_dataset = WallDataset(test_image_dir, test_mask_dir)
        test_loader = DataLoader(
            test_dataset,
            batch_size=VALIDATION_CONFIG['batch_size'],
            shuffle=False,
            num_workers=VALIDATION_CONFIG['num_workers']
        )
        
        logger.info(f"Evaluating on {len(test_dataset)} test samples")
        
        all_metrics = {
            'iou': [],
            'dice': [],
            'accuracy': [],
            'precision': [],
            'recall': []
        }
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluating')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate metrics
                batch_metrics = SegmentationMetrics.calculate_all_metrics(
                    outputs.detach(),
                    masks.detach(),
                    threshold=0.5
                )
                
                for metric_name, metric_value in batch_metrics.items():
                    all_metrics[metric_name].append(metric_value)
        
        # Calculate averages
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)
        for metric_name, metric_value in avg_metrics.items():
            logger.info(f"{metric_name.upper():12s}: {metric_value:.4f}")
        logger.info("=" * 60)
        
        return avg_metrics


def main():
    """Example evaluation"""
    # Try to use best model if available, otherwise use quick_trained_model
    model_path = "models/best_wall_segmentation_model.pth"
    if not os.path.exists(model_path):
        model_path = "models/quick_trained_model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"No model found at {model_path}")
        logger.error("Please run training first: python quick_train.py or python train.py")
        return
    
    logger.info(f"Using model: {model_path}")
    evaluator = ModelEvaluator(model_path)
    
    # If test masks don't exist, use validation set instead
    test_image_dir = "data/test/images"
    test_mask_dir = "data/test/masks"
    
    if not os.path.exists(test_mask_dir) or len(os.listdir(test_mask_dir)) == 0:
        logger.info("No test masks found, using validation set instead")
        test_image_dir = "data/val/images"
        test_mask_dir = "data/val/masks"
    
    metrics = evaluator.evaluate(
        test_image_dir=test_image_dir,
        test_mask_dir=test_mask_dir
    )


if __name__ == "__main__":
    main()
