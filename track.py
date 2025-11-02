
from ultralytics import YOLO
import cv2
import os
import csv


def load_Model():
    model=YOLO("yolo11n.pt")
    return  model

def load_video():
    video_path='/home/terex/Downloads/4.mp4'
    return video_path

def track(model,frame):
    result=model.track(frame,persist=True)

    detect=result[0]
    box_id=detect.boxes.id.int()
    if box_id is not None:
        tracker_ids=box_id.cpu().tolist()

        for box,tracker_id in zip(detect.boxes,tracker_ids):

            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

        class_ids=int(box.cls.item())
        class_name=detect.names[class_ids]
        confidence=float(box.conf.item())

        label=f"ID: {tracker_id}{class_ids}{class_name}{confidence:.2f}"
        log_to_csv("videoData.csv",header = ['Frame', 'Track_ID', 'Class', 'x1', 'y1', 'x2', 'y2'],row = [frame_number, track_id, class_name, x1, y1, x2, y2])
        cv2.reactangle(frame,(x1,y1),(x2,y2),(0,245.0),2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame




def log_to_csv(csv_filename, csv_header, data_row):


    # Check if we need to write the header
    write_header = False

    if not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0:
        write_header = True

    try:

        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(csv_header)  # Write header

            writer.writerow(data_row)  # Write the data

    except Exception as e:
        print(f"Error logging to CSV: {e}")

if __name__ == '__main__':

    model=load_Model()

    cap=cv2.VideoCapture(load_video())

    while True:
        success,frame=cap.read()
        if not success:
             break


             cv2.imshow("YOLOv8 Tracking", track(load_model,frame))

             # Break the loop if 'q' is pressed
             if cv2.waitKey(1) & 0xFF == ord("q"):
                 break

cap.release()
cv2.destroyAllWindows()