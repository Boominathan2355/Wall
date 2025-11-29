"""Module imports and utilities"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import logging

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Setup basic logging
logging.basicConfig(level=logging.INFO)
