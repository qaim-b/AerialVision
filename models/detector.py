"""
AerialVision Detector - YOLOv11 for Traffic Monitoring
Optimized for aerial vehicle detection
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
from ultralytics import YOLO
import time


class TrafficDetector:
    """
    YOLOv11-based vehicle detector optimized for aerial traffic imagery
    """
    
    # Vehicle class mapping for traffic monitoring
    VEHICLE_CLASSES = {
        'car': 'Car',
        'van': 'Van',
        'truck': 'Truck',
        'bus': 'Bus',
        'motor': 'Motorcycle',
        'bicycle': 'Bicycle'
    }
    
    def __init__(
        self,
        model_path: str = 'yolo11n.pt',
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        img_size: int = 640
    ):
        """
        Initialize traffic detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device for inference
            img_size: Input image size
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Warm up model
        self._warmup()
        
        print(f"Detector ready on {self.device}")
    
    def _warmup(self):
        """Warm up model with dummy input"""
        dummy = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy, verbose=False)
    
    def detect(
        self,
        image: Union[str, np.ndarray],
        return_image: bool = True
    ) -> Dict:
        """
        Detect vehicles in image
        
        Args:
            image: Image path or numpy array
            return_image: Whether to return annotated image
            
        Returns:
            Detection results dictionary
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False
        )[0]
        
        inference_time = time.time() - start_time
        
        # Process detections
        detections = self._process_results(results)
        
        # Count by vehicle type
        vehicle_counts = self._count_by_class(detections)
        
        response = {
            'detections': detections,
            'count': len(detections),
            'total_vehicles': sum(vehicle_counts.values()),
            'by_class': vehicle_counts,
            'inference_time': round(inference_time, 3),
            'image_size': list(results.orig_shape),
            'device': self.device
        }
        
        if return_image:
            annotated = results.plot()
            response['annotated_image'] = annotated
        
        return response
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False
    ) -> Dict:
        """
        Detect vehicles in video
        
        Args:
            video_path: Input video path
            output_path: Output video path
            show: Display video
            
        Returns:
            Video processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        inference_times = []
        all_vehicle_counts = []
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            result = self.detect(frame, return_image=True)
            
            frame_count += 1
            total_detections += result['count']
            inference_times.append(result['inference_time'])
            all_vehicle_counts.append(result['total_vehicles'])
            
            # Write/show frame
            annotated_frame = result['annotated_image']
            
            if writer:
                writer.write(annotated_frame)
            
            if show:
                cv2.imshow('Traffic Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        avg_inference = np.mean(inference_times)
        avg_fps = 1.0 / avg_inference if avg_inference > 0 else 0
        avg_vehicles = np.mean(all_vehicle_counts)
        
        return {
            'frames_processed': frame_count,
            'total_detections': total_detections,
            'avg_vehicles_per_frame': round(avg_vehicles, 2),
            'avg_inference_time': round(avg_inference, 3),
            'avg_fps': round(avg_fps, 2),
            'output_path': output_path
        }
    
    def _process_results(self, results) -> List[Dict]:
        """Process YOLO results into standard format"""
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = results.names[cls_id]
                
                detection = {
                    'class_id': int(cls_id),
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x) for x in box],
                    'bbox_center': [
                        float((box[0] + box[2]) / 2),
                        float((box[1] + box[3]) / 2)
                    ],
                    'bbox_area': float((box[2] - box[0]) * (box[3] - box[1]))
                }
                detections.append(detection)
        
        return detections
    
    def _count_by_class(self, detections: List[Dict]) -> Dict[str, int]:
        """Count detections by vehicle class"""
        counts = {
            'car': 0,
            'van': 0,
            'truck': 0,
            'bus': 0,
            'motor': 0,
            'bicycle': 0,
            'other': 0
        }
        
        for det in detections:
            class_name = det['class_name'].lower()
            if class_name in counts:
                counts[class_name] += 1
            else:
                counts['other'] += 1
        
        return counts
    
    def update_thresholds(self, conf: Optional[float] = None, iou: Optional[float] = None):
        """Update detection thresholds"""
        if conf is not None:
            self.conf_threshold = conf
        if iou is not None:
            self.iou_threshold = iou


if __name__ == '__main__':
    # Test detector
    detector = TrafficDetector(model_path='yolov11n.pt')
    
    # Test on dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    result = detector.detect(dummy_image)
    print(f"Test completed: {result['total_vehicles']} vehicles detected")
    print(f"Inference time: {result['inference_time']}s")
