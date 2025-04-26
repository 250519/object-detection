import cv2
import numpy as np
import threading
from ultralytics import YOLO

# --- Configurations ---
CONFIDENCE_THRESHOLD = 0.6
FRAME_HISTORY = 60
NMS_IOU_THRESHOLD = 0.5
IOU_MATCH_THRESHOLD = 0.45


# --- Initialize ---
model = YOLO("yolov8n.pt")

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

object_db = {}  # ObjectID -> (class_name, bbox, last_seen_frame)
object_counter = 0
frame_count = 0

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

print("[INFO] Starting YOLOv8 object monitoring with threaded video stream and IoU tracking...")

while True:
    frame = vs.read()
    if frame is None:
        break

    frame_count += 1
    results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, iou=NMS_IOU_THRESHOLD, verbose=False)[0]

    current_objects = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        current_objects.append((class_name, (x1, y1, x2, y2)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    matched_ids = set()

    for class_curr, bbox_curr in current_objects:
        already_known = False
        for oid, (class_db, bbox_db, last_seen) in object_db.items():
            if class_curr == class_db and calculate_iou(bbox_curr, bbox_db) > IOU_MATCH_THRESHOLD:
                object_db[oid] = (class_db, bbox_curr, frame_count)
                already_known = True
                matched_ids.add(oid)
                break

        if not already_known:
            object_counter += 1
            object_db[object_counter] = (class_curr, bbox_curr, frame_count)
            print(f"[NEW OBJECT] ID {object_counter} ({class_curr}) detected at frame {frame_count}")

    to_delete = []
    for oid, (class_db, bbox_db, last_seen) in object_db.items():
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