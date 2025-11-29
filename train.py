"""Training script for wall segmentation model"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm
from datetime import datetime

from import_modules import device
from wallsegmentation import UNetWallSegmenter
from dataset import WallDataset
from losses_and_metrics import CombinedLoss, SegmentationMetrics
from config import TRAINING_CONFIG, VALIDATION_CONFIG, MODEL_CONFIG, PATHS

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer class for wall segmentation model"""
    
    def __init__(self, model_path=None):
        """
        Initialize trainer
        
        Args:
            model_path (str): Path to pretrained model. If None, creates new model.
        """
        self.device = device
        self.model = UNetWallSegmenter(
            in_channels=MODEL_CONFIG['in_channels'],
            out_channels=MODEL_CONFIG['out_channels']
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded pretrained model from {model_path}")
        
        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0
        metrics_list = {'iou': [], 'dice': [], 'accuracy': []}
        
        pbar = tqdm(train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = SegmentationMetrics.calculate_all_metrics(
                    outputs.detach(),
                    masks.detach(),
                    threshold=0.5
                )
                for metric_name, metric_value in metrics.items():
                    if metric_name in metrics_list:
                        metrics_list[metric_name].append(metric_value)
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in metrics_list.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate(self, val_loader):
        """
        Validate model
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        total_loss = 0
        metrics_list = {'iou': [], 'dice': [], 'accuracy': []}
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # Calculate metrics
                metrics = SegmentationMetrics.calculate_all_metrics(
                    outputs.detach(),
                    masks.detach(),
                    threshold=0.5
                )
                for metric_name, metric_value in metrics.items():
                    if metric_name in metrics_list:
                        metrics_list[metric_name].append(metric_value)
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in metrics_list.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch
            metrics (dict): Validation metrics
            is_best (bool): Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, PATHS['model_checkpoint'])
        
        # Save best model
        if is_best:
            torch.save(checkpoint, PATHS['best_model'])
            logger.info(f"Best model saved with val_loss: {metrics['loss']:.4f}")
    
    def train(self, train_image_dir, train_mask_dir, val_image_dir, val_mask_dir,
              epochs=None, batch_size=None):
        """
        Complete training loop
        
        Args:
            train_image_dir (str): Path to training images
            train_mask_dir (str): Path to training masks
            val_image_dir (str): Path to validation images
            val_mask_dir (str): Path to validation masks
            epochs (int): Number of epochs. Uses config if None.
            batch_size (int): Batch size. Uses config if None.
        """
        if epochs is None:
            epochs = TRAINING_CONFIG['epochs']
        if batch_size is None:
            batch_size = TRAINING_CONFIG['batch_size']
        
        logger.info("=" * 80)
        logger.info("Starting Model Training")
        logger.info("=" * 80)
        logger.info(f"Training started at {datetime.now()}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Device: {self.device}")
        
        # Create datasets and dataloaders
        train_dataset = WallDataset(train_image_dir, train_mask_dir)
        val_dataset = WallDataset(val_image_dir, val_mask_dir)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=TRAINING_CONFIG['num_workers']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=VALIDATION_CONFIG['batch_size'],
            shuffle=False,
            num_workers=VALIDATION_CONFIG['num_workers']
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_metrics'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_metrics'].append(val_metrics)
            
            # Log metrics
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Train IoU: {train_metrics.get('iou', 0):.4f} | "
                f"Val IoU: {val_metrics.get('iou', 0):.4f}"
            )
            
            # Step scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
        
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Best model saved to {PATHS['best_model']}")
        logger.info("=" * 80)


def main():
    """Example training script"""
    trainer = ModelTrainer()
    
    # Update paths to your data directories
    trainer.train(
        train_image_dir="data/train/images",
        train_mask_dir="data/train/masks",
        val_image_dir="data/val/images",
        val_mask_dir="data/val/masks",
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size']
    )


if __name__ == "__main__":
    main()
