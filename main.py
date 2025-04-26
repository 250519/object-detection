import cv2
import numpy as np
import threading
from ultralytics import YOLO

# --- Configurations ---
CONFIDENCE_THRESHOLD = 0.5
FRAME_HISTORY = 30

# --- Initialize ---
model = YOLO("yolov8n.pt")  # Using a small model for real-time

# --- Threaded Camera Capture ---
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

vs = VideoStream(src=0).start()

object_db = {}  # ObjectID -> (class_name, centroid, last_seen_frame)
object_counter = 0
frame_count = 0

def get_centroid(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def match_objects(current_objects, object_db, threshold=50):
    matched_ids = set()
    for oid, (class_db, centroid_db, last_seen) in object_db.items():
        for class_curr, centroid_curr in current_objects:
            if class_db == class_curr and np.linalg.norm(np.array(centroid_db) - np.array(centroid_curr)) < threshold:
                matched_ids.add(oid)
    return matched_ids

print("[INFO] Starting YOLOv8 object monitoring with threaded video stream...")

while True:
    frame = vs.read()
    if frame is None:
        break

    frame_count += 1
    results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    current_objects = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centroid = get_centroid(x1, y1, x2, y2)

        current_objects.append((class_name, centroid))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, centroid, 5, (255, 0, 0), -1)

    # Match current detections with database
    matched_ids = match_objects(current_objects, object_db)

    # Update seen objects
    for class_curr, centroid_curr in current_objects:
        already_known = False
        for oid, (class_db, centroid_db, last_seen) in object_db.items():
            if class_curr == class_db and np.linalg.norm(np.array(centroid_db) - np.array(centroid_curr)) < 50:
                object_db[oid] = (class_db, centroid_curr, frame_count)
                already_known = True
                break

        if not already_known:
            object_counter += 1
            object_db[object_counter] = (class_curr, centroid_curr, frame_count)
            print(f"[NEW OBJECT] ID {object_counter} ({class_curr}) detected at frame {frame_count}")

    # Check for disappeared objects
    to_delete = []
    for oid, (class_db, centroid_db, last_seen) in object_db.items():
        if frame_count - last_seen > FRAME_HISTORY:
            print(f"[MISSING OBJECT] ID {oid} ({class_db}) disappeared at frame {frame_count}")
            to_delete.append(oid)

    for oid in to_delete:
        del object_db[oid]

    cv2.imshow('YOLOv8 Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
