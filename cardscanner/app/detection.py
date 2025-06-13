"""Card detection using YOLOv8."""
from pathlib import Path
from typing import Tuple, Optional

import cv2
from ultralytics import YOLO


class CardDetector:
    """Wrapper around YOLOv8 model for card detection."""

    def __init__(self, model_path: str | Path):
        self.model = YOLO(str(model_path))

    def detect(self, image_path: str | Path) -> Optional[Tuple[int, int, int, int]]:
        """Detect card bounding box from an image.

        Returns (x1, y1, x2, y2) or None if not found.
        """
        results = self.model.predict(str(image_path), conf=0.25)
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        # Take the largest bounding box
        box = max(boxes, key=lambda b: b.area)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        return x1, y1, x2, y2
