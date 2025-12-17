from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# config
cat_class_id = 15
box_color = (255, 0, 0)
name_text = "Piyawat-Clicknext-Internship-2024"

track_points = []

def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    global track_points

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]

        if class_id != cat_class_id:
            continue

        coordinator = box.xyxy[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # tracking center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        track_points.append((cx, cy))

        if len(track_points) > 1:
            track_points.pop(0)

        # Draw bounding box
        annotator.box_label(
            box=coordinator, 
            label=class_name, 
            color=box_color
        )

    frame = annotator.result()

    for i in range(1, len(track_points)):
        cv.line(frame, track_points[i - 1], track_points[i], box_color, 2)

    return frame


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model(frame, conf=0.37)

    for result in results:
        frame = draw_boxes(frame, result.boxes)

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()

        if not ret:
            break

        if ret:
            # Detect motorcycle from image frame
            frame_result = detect_object(frame)

            # Draw name text
            test_size, _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x_text = frame_result.shape[1] - test_size[0] - 10
            y_text = 30

            cv2.putText(
                frame_result, 
                name_text, 
                (x_text, y_text), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                box_color, 
                2
            )

            # Show result
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow("Video", frame_result)
            cv2.waitKey(30)

        else:
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()
