# 🧍‍♂️ Player Re-Identification — Single Video Feed

## 📌 Overview

This repository contains a pipeline to identify and track **football players** across a **single video feed**. Each player is assigned a **unique, persistent ID** that remains consistent across frames using YOLOv11 and Deep SORT.

---

## 🗂️ Folder Structure

PlayerReID/
├── input.mp4 # Input video (15 seconds, 375 frames)
├── output.mp4 # Output with tracked player IDs
├── detect_and_track.py # Main pipeline (YOLO + Deep SORT)
├── reid_utils.py # Optional: ReID embedding module
├── requirements.txt # Python dependencies
├── test_read_video.py # (Optional) Frame test script
├── test_frame.jpg # (Optional) Frame image
├── report.md # Technical report
└── README.md # Setup instructions

yaml
Copy
Edit

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/PlayerReID.git
cd PlayerReID

2. Install Dependencies

pip install -r requirements.txt
🔧 requirements.txt

▶️ How to Run
Place your video as input.mp4 in the root folder.

Ensure your trained YOLOv11 weights are saved as best.pt.

Run the detection & tracking script:
python detect_and_track.py

Output will be saved to output.mp4.

🎯 Model Components
YOLOv11: Player detection in every frame.

Deep SORT: Tracks and re-identifies players across frames.

Optional: reid_utils.py can be used for embedding-level identity matching.

📝 Notes
Re-identification is done using Deep SORT motion-based tracking.

The script prints logs for every frame, detection, and player ID assignment.

You can enhance the tracking quality by integrating the unused reid_utils.py.

📌 Final Notes

- The output video file (`output.mp4`) was verified and plays correctly in external media players.
- Due to limitations of VS Code, the video may not open directly inside the IDE. This is a known behavior and not a bug.


