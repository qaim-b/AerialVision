"""
DeepSORT Tracker for Multi-Vehicle Tracking
Tracks vehicles across video frames
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import deque
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class Track:
    """Single tracked vehicle"""
    
    next_id = 1
    
    def __init__(self, detection: np.ndarray, class_id: int, confidence: float):
        self.track_id = Track.next_id
        Track.next_id += 1
        
        self.class_id = class_id
        self.confidence = confidence
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        
        # Kalman filter for state estimation
        self.kf = self._init_kalman(detection)
        
        # Track history
        self.history = deque(maxlen=30)
        self.history.append(detection)
    
    def _init_kalman(self, detection: np.ndarray) -> KalmanFilter:
        """Initialize Kalman filter for tracking"""
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Noise parameters
        kf.R *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        x, y, w, h = self._bbox_to_state(detection)
        kf.x[:4] = [[x], [y], [w], [h]]
        
        return kf
    
    def _bbox_to_state(self, bbox: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert bbox [x1,y1,x2,y2] to [cx,cy,w,h]"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return cx, cy, w, h
    
    def _state_to_bbox(self) -> np.ndarray:
        """Convert state [cx,cy,w,h] back to [x1,y1,x2,y2]"""
        cx, cy, w, h = self.kf.x[:4].flatten()
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])
    
    def predict(self):
        """Predict next state"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection: np.ndarray, confidence: float):
        """Update track with new detection"""
        self.time_since_update = 0
        self.hits += 1
        self.confidence = confidence
        
        # Update Kalman filter
        x, y, w, h = self._bbox_to_state(detection)
        measurement = np.array([[x], [y], [w], [h]])
        self.kf.update(measurement)
        
        # Update history
        self.history.append(detection)
    
    def get_state(self) -> np.ndarray:
        """Get current bbox estimate"""
        return self._state_to_bbox()
    
    def get_track_info(self) -> Dict:
        """Get track information"""
        bbox = self.get_state()
        return {
            'track_id': self.track_id,
            'bbox': bbox.tolist(),
            'class_id': self.class_id,
            'confidence': self.confidence,
            'age': self.age,
            'hits': self.hits
        }


class VehicleTracker:
    """
    DeepSORT-based multi-vehicle tracker
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracker
        
        Args:
            max_age: Max frames to keep track without updates
            min_hits: Min hits before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Predict new locations
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        if len(detections) > 0:
            det_boxes = np.array([d['bbox'] for d in detections])
            track_boxes = np.array([t.get_state() for t in self.tracks]) if self.tracks else np.empty((0, 4))
            
            matched, unmatched_dets, unmatched_tracks = self._associate(det_boxes, track_boxes)
            
            # Update matched tracks
            for det_idx, track_idx in matched:
                self.tracks[track_idx].update(
                    det_boxes[det_idx],
                    detections[det_idx]['confidence']
                )
            
            # Create new tracks
            for det_idx in unmatched_dets:
                new_track = Track(
                    det_boxes[det_idx],
                    detections[det_idx]['class_id'],
                    detections[det_idx]['confidence']
                )
                self.tracks.append(new_track)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        # Return confirmed tracks
        confirmed = [
            t.get_track_info() for t in self.tracks
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits
        ]
        
        return confirmed
    
    def _associate(
        self,
        detections: np.ndarray,
        tracks: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using IoU"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(detections, tracks)
        
        # Hungarian algorithm
        if iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(range(len(tracks)))
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] > self.iou_threshold:
                    matched.append((r, c))
                    unmatched_dets.remove(r)
                    unmatched_tracks.remove(c)
            
            return matched, unmatched_dets, unmatched_tracks
        
        return [], list(range(len(detections))), list(range(len(tracks)))
    
    def _compute_iou_matrix(self, boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of boxes"""
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        
        iou_matrix = np.zeros((len(boxes_a), len(boxes_b)))
        
        for i, box_a in enumerate(boxes_a):
            for j, box_b in enumerate(boxes_b):
                # Intersection
                x1 = max(box_a[0], box_b[0])
                y1 = max(box_a[1], box_b[1])
                x2 = min(box_a[2], box_b[2])
                y2 = min(box_a[3], box_b[3])
                
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                union_area = area_a[i] + area_b[j] - inter_area
                
                iou_matrix[i, j] = inter_area / union_area if union_area > 0 else 0
        
        return iou_matrix
    
    def reset(self):
        """Reset tracker"""
        self.tracks = []
        self.frame_count = 0
        Track.next_id = 1


if __name__ == '__main__':
    # Test tracker
    tracker = VehicleTracker()
    
    # Simulate detections
    dummy_detections = [
        {'bbox': [100, 100, 200, 200], 'class_id': 0, 'confidence': 0.9}
    ]
    
    tracks = tracker.update(dummy_detections)
    print(f"Test completed: {len(tracks)} tracks")
