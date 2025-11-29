"""Configuration file for wall detection system"""

# Model Configuration
MODEL_CONFIG = {
    'in_channels': 3,
    'out_channels': 1,
    'image_size': 512,
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 0,  # Change to 2-4 on Linux/Mac
}

# Validation Configuration
VALIDATION_CONFIG = {
    'batch_size': 8,
    'num_workers': 0,
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'image_size': 512,
    'horizontal_flip': 0.5,
    'brightness_contrast': 0.2,
    'rotate': 0.3,
    'scale': 0.2,
}

# Inference Configuration
INFERENCE_CONFIG = {
    'confidence_threshold': 0.5,
    'min_wall_area': 100,  # Minimum pixels for a wall to be detected
    'scale_factor': 100,   # pixels per meter
}

# Unit Conversion Constants
UNIT_CONVERSION = {
    'meters_to_feet': 3.28084,
    'sqm_to_sqft': 10.764,  # 1 square meter = 10.764 square feet
}

# Paths
PATHS = {
    'model_checkpoint': 'models/wall_segmentation_model.pth',
    'best_model': 'models/best_wall_segmentation_model.pth',
    'training_log': 'logs/training.log',
}

# Device
DEVICE = 'cuda'  # or 'cpu'
