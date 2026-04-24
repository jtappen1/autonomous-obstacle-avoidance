import cv2
import time
import os

save_dir = "dataset_frames"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture('/dev/video4', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

frame_idx = 0
save_every = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # save raw frame
    if frame_idx % save_every == 0:
        cv2.imwrite(f"{save_dir}/frame_{frame_idx:06d}.jpg", frame)

    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()