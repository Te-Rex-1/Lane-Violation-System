from ultralytics import YOLO
import cv2
model=YOLO('yolo11n.pt')
class_list=model.names

print(class_list)
video_path='/home/terex/Downloads/4.mp4'

cap=cv2.VideoCapture(video_path)


frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "output_detected.mp4",             # output filename
    cv2.VideoWriter_fourcc(*"mp4v"),   # codec
    fps,                               # frames per second
    (frame_width, frame_height)        # frame size
)



while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    detections=results[0]

    for box in detections.boxes:
        xyxy = box.xyxy[0]
        print(xyxy)
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

        class_id = int(box.cls.item())
        class_name = detections.names[class_id]

        confidence=float(box.conf.item())
        if class_id==2 and confidence>0.5:
            cv2.putText(frame,'bakri', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("YOLO Vehicle Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()