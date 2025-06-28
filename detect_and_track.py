import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random


video_path = "input.mp4"
model_path = "best.pt"
output_path = "output.mp4"

print("[👟] Script started...")

# Load YOLOv11 model
print("[ℹ️] Loading YOLOv11 model...")
try:
    model = YOLO(model_path)
    print("[✅] YOLO model loaded from", model_path)
except Exception as e:
    print("[❌] Failed to load YOLO model:", e)
    exit()

# Initialize Deep SORT tracker
print("[📦] Initializing Deep SORT tracker...")
tracker = DeepSort(max_age=30,
                   n_init=3,
                   nms_max_overlap=1.0,
                   max_cosine_distance=0.,
                   nn_budget=None,
                   override_track_class=None)
print("[✅] Deep SORT initialized")

# Open input video
print(f"[📂] Opening video file: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[❌] Failed to open video.")
    exit()
else:
    print("[✅] Video opened successfully")

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[ℹ️] Video FPS: {fps}, Total Frames: {frame_count}, Resolution: {width}x{height}")

# Prepare writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
print(f"[📝] Output will be saved to: {output_path}")

# Tracking variables
frame_idx = 0
id_colors = {}

def get_color_for_id(track_id):
    if track_id not in id_colors:
        id_colors[track_id] = tuple(random.randint(0, 255) for _ in range(3))
    return id_colors[track_id]

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("[⚠️] End of video or failed to read frame.")
        break

    print(f"\n[🔁] Processing Frame {frame_idx + 1}/{frame_count}")

    results = model.predict(frame, verbose=False)[0]
    print(f"[📦] Detected {len(results.boxes)} objects")

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "player"))

    print(f"[🔍] {len(detections)} detections passed to Deep SORT")

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        x1, y1, x2, y2 = int(l), int(t), int(r), int(b)
        color = get_color_for_id(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Player {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        print(f"[🧍] Player {track_id} @ ({x1},{y1},{x2},{y2})")

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

cv2.destroyAllWindows()
print("\n[✅] Video processing complete.")
print("[📁] Saved output to:", output_path)