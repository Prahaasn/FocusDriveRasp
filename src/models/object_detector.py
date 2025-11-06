"""
TensorFlow Lite Object Detection Module
Detects objects like phones, cups, bottles for distraction detection
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
import tensorflow as tf


class Detection:
    """Represents a detected object"""
    def __init__(self, class_id: int, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x, y, width, height)

    def __repr__(self):
        return f"Detection(class='{self.class_name}', conf={self.confidence:.2f}, bbox={self.bbox})"


class TFLiteObjectDetector:
    """
    TensorFlow Lite Object Detector for real-time distraction detection
    Uses MobileNet SSD model trained on COCO dataset
    """

    # COCO classes relevant for distraction detection
    RELEVANT_CLASSES = {
        'cell phone': {'id': 77, 'distraction_level': 'high'},
        'cup': {'id': 47, 'distraction_level': 'medium'},
        'bottle': {'id': 44, 'distraction_level': 'medium'},
        'wine glass': {'id': 46, 'distraction_level': 'medium'},
        'book': {'id': 84, 'distraction_level': 'medium'},
        'laptop': {'id': 73, 'distraction_level': 'high'},
        'person': {'id': 1, 'distraction_level': 'low'},
    }

    def __init__(self, model_path: str, labelmap_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the TFLite object detector

        Args:
            model_path: Path to .tflite model file
            labelmap_path: Path to labelmap.txt file
            confidence_threshold: Minimum confidence for detection (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.labels = self._load_labels(labelmap_path)

        # Initialize TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        print(f"TFLite Object Detector initialized:")
        print(f"  Model input size: {self.input_width}x{self.input_height}")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Tracking {len(self.RELEVANT_CLASSES)} relevant classes")

    def _load_labels(self, labelmap_path: str) -> Dict[int, str]:
        """Load COCO labels from labelmap file"""
        labels = {}
        with open(labelmap_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Format: "id name" or just "name"
                    parts = line.split(' ', 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        labels[int(parts[0])] = parts[1]
                    else:
                        # If no ID, use line number as ID
                        labels[len(labels)] = line
        return labels

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for TFLite model

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Preprocessed image ready for inference
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image_resized = cv2.resize(image_rgb, (self.input_width, self.input_height))

        # Add batch dimension and convert to uint8 (quantized model expects uint8)
        input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

        return input_data

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in image

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            List of Detection objects
        """
        # Preprocess image
        input_data = self._preprocess_image(image)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get results
        # Output format for MobileNet SSD:
        # boxes: [1, num_detections, 4] - bounding boxes (ymin, xmin, ymax, xmax) normalized to 0-1
        # classes: [1, num_detections] - class IDs
        # scores: [1, num_detections] - confidence scores
        # num_detections: [1] - number of valid detections

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # Bounding box coordinates
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # Class IDs
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  # Confidence scores

        # Parse detections
        detections = []
        img_height, img_width = image.shape[:2]

        for i in range(len(scores)):
            if scores[i] > self.confidence_threshold:
                # Get class info
                class_id = int(classes[i])
                class_name = self.labels.get(class_id, f"unknown_{class_id}")

                # Only keep relevant classes
                if class_name not in self.RELEVANT_CLASSES:
                    continue

                # Convert normalized coordinates to pixel coordinates
                ymin, xmin, ymax, xmax = boxes[i]
                x = int(xmin * img_width)
                y = int(ymin * img_height)
                w = int((xmax - xmin) * img_width)
                h = int((ymax - ymin) * img_height)

                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(scores[i]),
                    bbox=(x, y, w, h)
                )
                detections.append(detection)

        return detections

    def detect_relevant_objects(self, image: np.ndarray) -> Dict[str, List[Detection]]:
        """
        Detect objects and group by distraction level

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Dictionary with keys 'high', 'medium', 'low' containing detections
        """
        detections = self.detect(image)

        grouped = {
            'high': [],
            'medium': [],
            'low': []
        }

        for det in detections:
            level = self.RELEVANT_CLASSES.get(det.class_name, {}).get('distraction_level', 'low')
            grouped[level].append(det)

        return grouped

    def draw_detections(self, image: np.ndarray, detections: List[Detection],
                       color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes and labels on image

        Args:
            image: Input image (BGR format)
            detections: List of Detection objects
            color: BGR color for bounding boxes
            thickness: Line thickness

        Returns:
            Image with drawn detections
        """
        output = image.copy()

        for det in detections:
            x, y, w, h = det.bbox

            # Choose color based on distraction level
            level = self.RELEVANT_CLASSES.get(det.class_name, {}).get('distraction_level', 'low')
            if level == 'high':
                box_color = (0, 0, 255)  # Red
            elif level == 'medium':
                box_color = (0, 165, 255)  # Orange
            else:
                box_color = (0, 255, 0)  # Green

            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), box_color, thickness)

            # Draw label background
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x, y - label_size[1] - 10),
                         (x + label_size[0], y), box_color, -1)

            # Draw label text
            cv2.putText(output, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output


if __name__ == "__main__":
    # Test the detector
    import sys

    if len(sys.argv) < 2:
        print("Usage: python object_detector.py <image_path>")
        sys.exit(1)

    # Initialize detector
    detector = TFLiteObjectDetector(
        model_path="models/tflite/detect.tflite",
        labelmap_path="models/tflite/labelmap.txt",
        confidence_threshold=0.5
    )

    # Load and process image
    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)

    # Detect objects
    detections = detector.detect(image)

    print(f"\nFound {len(detections)} relevant objects:")
    for det in detections:
        print(f"  {det}")

    # Draw detections
    output = detector.draw_detections(image, detections)

    # Save result
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, output)
    print(f"\nResult saved to {output_path}")
