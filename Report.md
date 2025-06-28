# 🧍‍♂️ Player Re-Identification from a Single Video Feed

## 🎯 Assignment Objective

The goal was to build a player **re-identification (ReID)** system from a **single video feed**. Each player in a football match should be assigned a unique **fixed and persistent ID** that remains **unchanged** across all 375 frames — even if they temporarily leave and re-enter the frame.

---

## ⚙️ My Approach

### 1. **Detection with YOLOv11**
- Used `Ultralytics YOLOv11` (custom `best.pt` weights).
- Ran detection on each of the 375 frames to extract bounding boxes and confidence scores.

### 2. **Tracking with Deep SORT**
- Integrated `Deep SORT` (Deep Simple Online and Realtime Tracking).
- Supplied detection boxes as `[x, y, w, h]`, confidence, and class label `"player"`.
- Deep SORT uses motion and IOU-based metrics to persist IDs over time.

---

## 🧪 Techniques Attempted

| Technique | Description | Outcome |
|----------|-------------|---------|
| Simple ID Counter | Every detection gets a new ID incrementally | IDs changed frequently; not usable |
| Centroid/IOU Matching | Match players by bounding box overlap | Worked partially, broke on re-entry |
| ReID Embeddings (Cosine Similarity) | Used cropped player images to match features | Too slow on CPU, somewhat inconsistent |
| Deep SORT | Used motion + appearance heuristics | Best result: stable for ~70–80% of players |

> 🔁 `reid_utils.py` was created to support embedding-based ID matching but was not integrated into the final run due to time constraints.

---

## ⚠️ Challenges Faced

- **ID Reset**: When players exited and re-entered the frame, they were often assigned new IDs.
- **Camera Motion**: Any camera shake or pan disrupted ID matching.
- **Right Side Instability**: Players on the right side often had ID switches more frequently than those on the left.
- **Time Constraints**: Training a proper TorchReID model or fusing ReID with Deep SORT required more time.

---

## ✅ Final Outcome

- Detection and tracking are functional and stable for most players.
- IDs are **mostly persistent** for players who remain in the frame or move slowly.
- Still occasional **ID switches** during occlusion or rapid movement.
- `reid_utils.py` included as **future extension** for embedding-based matching.

---

## 🔮 Future Scope

- Train a **custom TorchReID model** on player crops to integrate with Deep SORT.
- Merge `ReID + Deep SORT` to ensure consistent tracking during occlusions and re-entries.
- Add **frame-level memory** to cache and retrieve previous identities using embeddings.
- Use **Kalman filtering or temporal smoothing** to reduce jitter in ID assignment.

---

## 📝 Submission Summary

- **Input**: Single video feed (`input.mp4`, 375 frames)
- **Output**: Annotated video (`output.mp4`) with consistent player IDs
- **Final Code Used**: `detect_and_track.py` with YOLOv11 + Deep SORT
- **Extra**: `reid_utils.py` module for future embedding-based enhancement


