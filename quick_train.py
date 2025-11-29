"""
Quick training script - simplified for faster training on sample data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WallDataset
from wallsegmentation import UNetWallSegmenter
from losses_and_metrics import DiceLoss
from config import TRAINING_CONFIG, MODEL_CONFIG
from import_modules import device
import os
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_train():
    """Quick training for demonstration"""
    
    logger.info("="*80)
    logger.info("QUICK TRAINING - WALL DETECTION SYSTEM")
    logger.info("="*80)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    model = UNetWallSegmenter(
        in_channels=MODEL_CONFIG['in_channels'],
        out_channels=MODEL_CONFIG['out_channels']
    ).to(device)
    
    # Create dataset and loader
    train_dataset = WallDataset(
        image_dir="data/train/images",
        mask_dir="data/train/masks"
    )
    
    # Use smaller batch size and fewer epochs for quick training
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    # Loss and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training - only 2 epochs
    num_epochs = 2
    
    logger.info(f"Device: {device}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Batch size: 2")
    logger.info(f"Epochs: {num_epochs}")
    logger.info("")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Save model
    model_path = 'models/quick_trained_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': num_epochs,
    }, model_path)
    logger.info(f"\n✓ Model saved to: {model_path}")
    
    logger.info("\n" + "="*80)
    logger.info("QUICK TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info("\nYou can now:")
    logger.info("1. Run evaluation: python evaluate.py")
    logger.info("2. Test on images: python usage.py")
    logger.info("3. Run full training: python train.py (for more epochs)")

if __name__ == "__main__":
    quick_train()
