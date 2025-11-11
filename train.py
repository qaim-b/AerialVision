"""
AerialVision Training Script
Train YOLOv11 on VisDrone dataset for traffic monitoring
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


def main():
    parser = argparse.ArgumentParser(description='Train AerialVision model')
    parser.add_argument('--data', type=str, default='data/visdrone.yaml',
                        help='Dataset YAML path')
    parser.add_argument('--weights', type=str, default='yolov11n.pt',
                        help='Initial weights path')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (0, 1, 2, ... or cpu)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Device: {args.device}\n")
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=50,
        save=True,
        save_period=10,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        verbose=True,
        plots=True
    )
    
    print(f"\nTraining complete!")
    print(f"Best model saved to: {results.save_dir}")
    print(f"Results:")
    print(f"  mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == '__main__':
    main()
