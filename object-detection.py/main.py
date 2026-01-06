import cv2
import time
import csv
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# CSV file setup
csv_file = open("detections.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Object", "Confidence"])

frame_count = 0
start_time = time.time()
total_confidence = 0
total_detections = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([timestamp, label, conf])

            total_confidence += conf
            total_detections += 1

    frame_count += 1
    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

end_time = time.time()
fps = frame_count / (end_time - start_time)
avg_conf = total_confidence / total_detections if total_detections else 0

# Save performance report
with open("performance_report.txt", "w") as f:
    f.write(f"Total Detections: {total_detections}\n")
    f.write(f"Average Confidence: {avg_conf:.2f}\n")
    f.write(f"FPS: {fps:.2f}\n")

cap.release()
csv_file.close()
cv2.destroyAllWindows()