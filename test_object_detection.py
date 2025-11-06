"""Quick test of TFLite object detection"""

import cv2
import sys
sys.path.insert(0, 'src')
from models.object_detector import TFLiteObjectDetector

# Initialize detector
print("Loading TFLite object detector...")
detector = TFLiteObjectDetector(
    model_path="models/tflite/detect.tflite",
    labelmap_path="models/tflite/labelmap.txt",
    confidence_threshold=0.5
)

# Test with webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    sys.exit(1)

print("Press 'q' to quit, 's' to save screenshot")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    detections = detector.detect(frame)

    # Draw detections
    output = detector.draw_detections(frame, detections)

    # Show counts
    cv2.putText(output, f"Objects detected: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display
    cv2.imshow("Object Detection Test", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("test_detection.jpg", output)
        print("Screenshot saved to test_detection.jpg")

cap.release()
cv2.destroyAllWindows()
print("Test complete!")
