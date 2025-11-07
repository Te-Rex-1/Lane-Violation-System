from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime


def load_Model():
    model=YOLO("yolo11n.pt")
    return  model

def load_video():
    video_path='/home/terex/Downloads/4.mp4'
    return video_path

def track(model, frame, csv_writer, counted_vehicle):
    result=model.track(frame,persist=True)
    detect=result[0]


    if detect.boxes.id is not None:
        tracker_ids=detect.boxes.id.int().cpu().tolist()

        for box,tracker_id in zip(detect.boxes,tracker_ids):

            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            class_ids=int(box.cls.item())
            class_name=detect.names[class_ids]
            confidence=float(box.conf.item())
            # to add unique vehicle id
            if tracker_id not in counted_vehicle:
                counted_vehicle.add(tracker_id)
                row = [datetime.now(), tracker_id, class_name, x1, y1, x2, y2]
                csv_writer.writerow(row)
                label=f"ID: {tracker_id} {class_name} {confidence:.2f}"

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,245,0),2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

if __name__ == '__main__':

    model=load_Model()

    cap=cv2.VideoCapture(load_video())

    # Generate CSV filename with a counter
    counter = 1


    while os.path.exists(f"videoData_{counter}.csv"):
        counter += 1
    csv_filename = f"videoData_{counter}.csv"
    csv_header = ['Timestamp', 'Track_ID', 'Class', 'x1', 'y1', 'x2', 'y2']

    counted_vehicle = set()


    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)  # Write header

        while True:
            success,frame=cap.read()

            if not success:
                break
            
            # Pass the writer to the track function
            processed_frame = track(model, frame, writer, counted_vehicle)
            cv2.imshow("YOLOv11Tracking", processed_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()