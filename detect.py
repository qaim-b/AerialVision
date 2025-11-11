"""
AerialVision Detection Script
Command-line interface for vehicle detection
"""

import argparse
from pathlib import Path
import cv2
from models.detector import TrafficDetector
from models.tracker import VehicleTracker


def main():
    parser = argparse.ArgumentParser(description='AerialVision - Vehicle Detection')
    parser.add_argument('--source', type=str, required=True,
                        help='Image, video, or webcam (0)')
    parser.add_argument('--weights', type=str, default='yolo11n.pt',
                        help='Model weights path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold')
    parser.add_argument('--track', action='store_true',
                        help='Enable tracking')
    parser.add_argument('--save', action='store_true',
                        help='Save results')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("Initializing detector...")
    detector = TrafficDetector(
        model_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Determine source type
    if args.source.isdigit():
        # Webcam
        process_webcam(detector, int(args.source), args)
    elif Path(args.source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Image
        process_image(detector, args.source, args, output_dir)
    elif Path(args.source).suffix.lower() in ['.mp4', '.avi', '.mov']:
        # Video
        process_video(detector, args.source, args, output_dir)
    else:
        print(f"Unsupported source: {args.source}")


def process_image(detector, image_path, args, output_dir):
    """Process single image"""
    print(f"Processing image: {image_path}")
    
    # Detect
    result = detector.detect(image_path, return_image=True)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Detection Results")
    print(f"{'='*50}")
    print(f"Total Vehicles: {result['total_vehicles']}")
    print(f"By Class:")
    for class_name, count in result['by_class'].items():
        if count > 0:
            print(f"  {class_name.capitalize()}: {count}")
    print(f"Inference Time: {result['inference_time']}s")
    print(f"{'='*50}\n")
    
    # Show
    if args.show:
        cv2.imshow('Detection', result['annotated_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save
    if args.save:
        output_path = output_dir / f"result_{Path(image_path).name}"
        cv2.imwrite(str(output_path), result['annotated_image'])
        print(f"Saved to: {output_path}")


def process_video(detector, video_path, args, output_dir):
    """Process video file"""
    print(f"Processing video: {video_path}")
    
    output_path = None
    if args.save:
        output_path = str(output_dir / f"result_{Path(video_path).name}")
    
    result = detector.detect_video(
        video_path,
        output_path=output_path,
        show=args.show
    )
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Video Processing Results")
    print(f"{'='*50}")
    print(f"Frames Processed: {result['frames_processed']}")
    print(f"Total Detections: {result['total_detections']}")
    print(f"Avg Vehicles/Frame: {result['avg_vehicles_per_frame']}")
    print(f"Avg Inference Time: {result['avg_inference_time']}s")
    print(f"Avg FPS: {result['avg_fps']}")
    if output_path:
        print(f"Saved to: {output_path}")
    print(f"{'='*50}\n")


def process_webcam(detector, camera_id, args):
    """Process webcam stream"""
    print(f"Opening webcam: {camera_id}")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Cannot open webcam {camera_id}")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        result = detector.detect(frame, return_image=True)
        
        # Display
        annotated = result['annotated_image']
        
        # Add stats overlay
        text = f"Vehicles: {result['total_vehicles']} | FPS: {1.0/result['inference_time']:.1f}"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 212, 170), 2)
        
        cv2.imshow('Webcam Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
