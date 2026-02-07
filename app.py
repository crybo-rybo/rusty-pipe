import cv2
from ultralytics import YOLO
import rust_cv_core

def main():
    # 1. Initialize YOLO model (using yolo11n for speed, it will auto-download)
    print("Loading YOLO model...")
    model = YOLO("yolo11n.pt") 

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting video stream. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # 3. Run Inference
        # stream=True is generally faster for loops, but regular call is fine for simple app
        results = model(frame, verbose=False) 
        
        # 4. Bridge to Rust
        # Map YOLO boxes to Rust Detection structs
        rust_detections = []
        
        # results[0] contains the detections for the first (and only) image
        for box in results[0].boxes:
            # box.xywh returns center_x, center_y, width, height
            # box.cls is the class ID tensor
            # box.conf is the confidence tensor
            
            x, y, w, h = box.xywh[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            detection = rust_cv_core.Detection(
                cls_id, 
                conf, 
                (x, y, w, h)
            )
            rust_detections.append(detection)

        # 5. Process in Rust
        # This filters for Humans (Class 0) with > 0.5 confidence
        filtered_detections = rust_cv_core.process_frame(rust_detections)

        # 6. Visualize Results (Draw only what Rust sent back)
        for det in filtered_detections:
            x, y, w, h = det.bbox
            
            # Convert xywh (center) to xyxy (top-left, bottom-right) for OpenCV drawing
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            # Draw Green Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label
            label = f"Person: {det.conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 7. Display
        cv2.imshow('Rusty-Pipe: Python + Rust CV', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
