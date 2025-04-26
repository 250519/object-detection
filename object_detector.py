import cv2
import numpy as np
from ultralytics import YOLO
from utils import draw_bounding_box, get_centroid

class SceneChangeDetector:
    """
    Detector class that processes video frames using YOLOv8 for object detection.
    """
    
    def __init__(self, min_confidence=0.5, stabilization_frames=30):
        """
        Initialize the detector with configuration parameters.
        
        Args:
            min_confidence: Minimum confidence threshold for YOLOv8 detections
            stabilization_frames: Number of frames to wait before detection starts
        """
        self.min_confidence = min_confidence
        self.stabilization_frames = stabilization_frames
        self.frame_count = 0
        
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Using the nano model for speed
        
        # Object tracking variables
        self.objects = {}  # Dictionary to track objects {id: {"bbox": (x,y,w,h), "class": class_name, "last_seen": frame_count}}
        self.reference_objects = {}  # Reference objects from the initial stable scene
        self.next_object_id = 1
        
        # Constants
        self.MAX_DISAPPEARED_FRAMES = 10
    
    def reset(self):
        """Reset the detector state."""
        self.frame_count = 0
        self.objects = {}
        self.reference_objects = {}
        self.next_object_id = 1
    
    def process_frame(self, frame):
        """
        Process a video frame to detect new and missing objects using YOLOv8.
        
        Args:
            frame: Input frame from video source
            
        Returns:
            tuple: (annotated_frame, events)
                annotated_frame: Frame with visual indicators
                events: List of detected events (new/missing objects)
        """
        self.frame_count += 1
        events = []
        
        # Create a copy of the frame for drawing
        result_frame = frame.copy()
        
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False)
        current_objects = {}
        
        # Before stabilization period, just accumulate object info
        if self.frame_count <= self.stabilization_frames:
            # Draw stabilization progress
            progress = int((self.frame_count / self.stabilization_frames) * 100)
            cv2.putText(result_frame, f"Stabilizing: {progress}%", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # If this is the last stabilization frame, record objects as reference
            if self.frame_count == self.stabilization_frames:
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        if box.conf.item() >= self.min_confidence:
                            bbox = box.xyxy[0].cpu().numpy()  # get box coordinates
                            class_id = int(box.cls.item())
                            class_name = r.names[class_id]
                            confidence = box.conf.item()
                            
                            # Convert bbox to (x, y, w, h) format
                            x1, y1, x2, y2 = map(int, bbox)
                            w = x2 - x1
                            h = y2 - y1
                            
                            # Register as a reference object
                            self.reference_objects[self.next_object_id] = {
                                "bbox": (x1, y1, w, h),
                                "class": class_name,
                                "confidence": confidence,
                                "last_seen": self.frame_count
                            }
                            self.next_object_id += 1
        else:
            # Process detections after stabilization
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if box.conf.item() >= self.min_confidence:
                        bbox = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls.item())
                        class_name = r.names[class_id]
                        confidence = box.conf.item()
                        
                        # Convert bbox to (x, y, w, h) format
                        x1, y1, x2, y2 = map(int, bbox)
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Check if this detection matches any existing object
                        object_id = self._match_object((x1, y1, w, h))
                        
                        if object_id is None:
                            # This is a new object
                            object_id = self.next_object_id
                            self.next_object_id += 1
                            events.append(("new", object_id, (x1, y1), class_name))
                        
                        # Update the object data
                        current_objects[object_id] = {
                            "bbox": (x1, y1, w, h),
                            "class": class_name,
                            "confidence": confidence,
                            "last_seen": self.frame_count
                        }
                        
                        # Draw the detection on the result frame
                        color = (0, 255, 0) if object_id in self.reference_objects else (0, 0, 255)
                        label = f"{class_name} {object_id} ({confidence:.2f})"
                        cv2.rectangle(result_frame, (x1, y1), (x1 + w, y1 + h), color, 2)
                        cv2.putText(result_frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Check for missing objects
            for obj_id, obj_data in self.reference_objects.items():
                if obj_id not in current_objects:
                    if obj_id in self.objects and (self.frame_count - self.objects[obj_id]["last_seen"]) >= self.MAX_DISAPPEARED_FRAMES:
                        x, y, w, h = obj_data["bbox"]
                        events.append(("missing", obj_id, (x, y), obj_data["class"]))
                        
                        # Draw missing object indicator
                        cv2.circle(result_frame, (x + w//2, y + h//2), 20, (0, 0, 255), 2)
                        cv2.line(result_frame, (x + w//2 - 15, y + h//2 - 15), 
                               (x + w//2 + 15, y + h//2 + 15), (0, 0, 255), 2)
                        cv2.line(result_frame, (x + w//2 + 15, y + h//2 - 15), 
                               (x + w//2 - 15, y + h//2 + 15), (0, 0, 255), 2)
                        cv2.putText(result_frame, f"Missing {obj_data['class']} {obj_id}", 
                                  (x + w//2 - 20, y + h//2 - 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update object tracking
        self.objects = current_objects
        
        # Draw stabilization notice if applicable
        if self.frame_count <= self.stabilization_frames:
            cv2.putText(result_frame, "Calibrating scene baseline...", 
                       (10, result_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Draw mode information
        cv2.putText(result_frame, "Mode: YOLOv8 Object Detection", 
                   (10, result_frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        return result_frame, events
    
    def _match_object(self, bbox, iou_threshold=0.5):
        """
        Match a bbox to an existing object based on IoU (Intersection over Union).
        
        Args:
            bbox: The (x, y, w, h) coordinates of the detection
            iou_threshold: Minimum IoU to consider a match
            
        Returns:
            object_id or None if no match found
        """
        if not self.objects:
            return None
        
        x1, y1, w1, h1 = bbox
        max_iou = 0
        best_match = None
        
        for obj_id, obj_data in self.objects.items():
            x2, y2, w2, h2 = obj_data["bbox"]
            
            # Calculate IoU
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)
            
            if inter_x2 < inter_x1 or inter_y2 < inter_y1:
                continue
            
            intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union = (w1 * h1) + (w2 * h2) - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou > max_iou:
                max_iou = iou
                best_match = obj_id
        
        return best_match if max_iou >= iou_threshold else None