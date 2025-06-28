import cv2

video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[❌] Could not open video:", video_path)
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[✅] Video loaded → Frames: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("[⛔] End of video.")
        break

    cv2.imshow("Frame Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print("[✅] Test read finished.")
