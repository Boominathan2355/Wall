"""Loss functions and evaluation metrics for wall segmentation"""
import torch
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Calculate Dice Loss
        
        Args:
            predictions (torch.Tensor): Model predictions [B, C, H, W]
            targets (torch.Tensor): Ground truth masks [B, C, H, W]
            
        Returns:
            torch.Tensor: Dice loss value
        """
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (predictions * targets).sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        return dice_loss


class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def forward(self, predictions, targets):
        """Combined loss"""
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class SegmentationMetrics:
    """Evaluation metrics for segmentation tasks"""
    
    @staticmethod
    def iou(predictions, targets, threshold=0.5):
        """
        Calculate Intersection over Union (IoU)
        
        Args:
            predictions (np.ndarray or torch.Tensor): Model predictions
            targets (np.ndarray or torch.Tensor): Ground truth
            threshold (float): Threshold for binary classification
            
        Returns:
            float: IoU score (0-1)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Binarize
        pred_binary = (predictions > threshold).astype(np.float32)
        target_binary = (targets > threshold).astype(np.float32)
        
        # Calculate
        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary) - intersection
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    @staticmethod
    def dice_coefficient(predictions, targets, threshold=0.5):
        """
        Calculate Dice Coefficient (F1 Score)
        
        Args:
            predictions (np.ndarray or torch.Tensor): Model predictions
            targets (np.ndarray or torch.Tensor): Ground truth
            threshold (float): Threshold for binary classification
            
        Returns:
            float: Dice coefficient (0-1)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Binarize
        pred_binary = (predictions > threshold).astype(np.float32)
        target_binary = (targets > threshold).astype(np.float32)
        
        # Calculate
        intersection = np.sum(pred_binary * target_binary)
        total = np.sum(pred_binary) + np.sum(target_binary)
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * intersection / total
    
    @staticmethod
    def accuracy(predictions, targets, threshold=0.5):
        """
        Calculate pixel-wise accuracy
        
        Args:
            predictions (np.ndarray or torch.Tensor): Model predictions
            targets (np.ndarray or torch.Tensor): Ground truth
            threshold (float): Threshold for binary classification
            
        Returns:
            float: Accuracy (0-1)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Binarize
        pred_binary = (predictions > threshold).astype(np.float32)
        target_binary = (targets > threshold).astype(np.float32)
        
        # Calculate
        correct = np.sum(pred_binary == target_binary)
        total = pred_binary.size
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def precision(predictions, targets, threshold=0.5):
        """
        Calculate precision (True Positives / (True Positives + False Positives))
        
        Args:
            predictions (np.ndarray or torch.Tensor): Model predictions
            targets (np.ndarray or torch.Tensor): Ground truth
            threshold (float): Threshold for binary classification
            
        Returns:
            float: Precision (0-1)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Binarize
        pred_binary = (predictions > threshold).astype(np.float32)
        target_binary = (targets > threshold).astype(np.float32)
        
        # Calculate
        tp = np.sum(pred_binary * target_binary)
        fp = np.sum(pred_binary * (1 - target_binary))
        
        if tp + fp == 0:
            return 0.0
        
        return tp / (tp + fp)
    
    @staticmethod
    def recall(predictions, targets, threshold=0.5):
        """
        Calculate recall (True Positives / (True Positives + False Negatives))
        
        Args:
            predictions (np.ndarray or torch.Tensor): Model predictions
            targets (np.ndarray or torch.Tensor): Ground truth
            threshold (float): Threshold for binary classification
            
        Returns:
            float: Recall (0-1)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Binarize
        pred_binary = (predictions > threshold).astype(np.float32)
        target_binary = (targets > threshold).astype(np.float32)
        
        # Calculate
        tp = np.sum(pred_binary * target_binary)
        fn = np.sum((1 - pred_binary) * target_binary)
        
        if tp + fn == 0:
            return 0.0
        
        return tp / (tp + fn)
    
    @staticmethod
    def calculate_all_metrics(predictions, targets, threshold=0.5):
        """
        Calculate all metrics at once
        
        Args:
            predictions (np.ndarray or torch.Tensor): Model predictions
            targets (np.ndarray or torch.Tensor): Ground truth
            threshold (float): Threshold for binary classification
            
        Returns:
            dict: Dictionary with all metrics
        """
        return {
            'iou': SegmentationMetrics.iou(predictions, targets, threshold),
            'dice': SegmentationMetrics.dice_coefficient(predictions, targets, threshold),
            'accuracy': SegmentationMetrics.accuracy(predictions, targets, threshold),
            'precision': SegmentationMetrics.precision(predictions, targets, threshold),
            'recall': SegmentationMetrics.recall(predictions, targets, threshold),
        }
