"""
Simple test script to verify installation and imports
Run this to check if all dependencies are installed correctly
"""
import sys


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...\n")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'albumentations': 'Albumentations',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
    }
    
    failed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} - OK")
        except ImportError:
            print(f"✗ {name:20s} - MISSING")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_local_modules():
    """Test if local modules can be imported"""
    print("\n\nTesting local modules...\n")
    
    modules = [
        'config',
        'import_modules',
        'wallsegmentation',
        'dataset',
        'losses_and_metrics',
        'wall_detection_system',
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module:25s} - OK")
        except Exception as e:
            print(f"✗ {module:25s} - ERROR: {str(e)[:50]}")
            failed.append(module)
    
    return len(failed) == 0, failed


def test_device():
    """Test GPU/CPU availability"""
    print("\n\nTesting device setup...\n")
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA Version: {torch.version.cuda}")
        else:
            print(f"✓ Using CPU (no GPU found)")
        
        print(f"✓ Device set to: {device}")
        return True, []
    except Exception as e:
        print(f"✗ Device check failed: {e}")
        return False, [str(e)]


def test_model():
    """Test model creation"""
    print("\n\nTesting model creation...\n")
    
    try:
        from wallsegmentation import UNetWallSegmenter
        from import_modules import device
        
        model = UNetWallSegmenter(in_channels=3, out_channels=1).to(device)
        print(f"✓ U-Net model created successfully")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Device: {device}")
        return True, []
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False, [str(e)]


def main():
    """Run all tests"""
    print("=" * 70)
    print("WALL DETECTION SYSTEM - INSTALLATION TEST")
    print("=" * 70)
    
    results = []
    
    # Test 1: Imports
    success, failed = test_imports()
    results.append(("Package Imports", success, failed))
    
    # Test 2: Local Modules
    success, failed = test_local_modules()
    results.append(("Local Modules", success, failed))
    
    # Test 3: Device
    success, failed = test_device()
    results.append(("Device Setup", success, failed))
    
    # Test 4: Model
    success, failed = test_model()
    results.append(("Model Creation", success, failed))
    
    # Summary
    print("\n\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, success, failed in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:25s} - {status}")
        if failed:
            for item in failed:
                print(f"  ├─ {item}")
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ All tests passed! System is ready to use.\n")
        print("Next steps:")
        print("  1. Prepare your dataset (images + binary masks)")
        print("  2. Run: python train.py")
        print("  3. Run: python evaluate.py")
        print("  4. Use: python usage.py or python webcam.py\n")
        return 0
    else:
        print("\n✗ Some tests failed. Please install missing packages.\n")
        print("Install with: pip install -r requirements.txt\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
