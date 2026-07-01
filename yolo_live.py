import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera not opened")
    exit()

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("outputs/output.avi", fourcc, 20.0, (1280, 720))

cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv8 Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3, iou=0.45, verbose=False)
    annotated = results[0].plot()

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", annotated)

    # SAVE VIDEO
    out.write(annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()