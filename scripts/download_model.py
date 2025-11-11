"""
Download pre-trained YOLOv11 model
"""

from ultralytics import YOLO
from pathlib import Path

def main():
    print("Downloading YOLOv11 model...")
    
    # This will automatically download the model if not present
    model = YOLO('yolo11n.pt')
    
    print(f"Model downloaded successfully")
    print(f"Model info:")
    print(f"  Name: YOLOv11n")
    print(f"  Parameters: ~2.6M")
    print(f"  Size: ~5.1 MB")
    print(f"  Speed: ~42 FPS (GTX 1660 Ti)")
    
    # Verify model
    print("\nVerifying model...")
    import torch
    import numpy as np
    
    dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    result = model(dummy_input, verbose=False)
    
    print("✓ Model verified and ready to use!")

if __name__ == '__main__':
    main()
