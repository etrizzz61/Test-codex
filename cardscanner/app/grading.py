"""Card condition estimation utilities."""
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np


class CardConditionEvaluator:
    """Estimate card condition from an image."""

    def evaluate(self, image_path: str | Path, bbox: Optional[Tuple[int, int, int, int]]) -> float:
        """Return a score between 0.0 (poor) and 1.0 (mint)."""
        if bbox is None:
            return 0.0
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        # Look at edge noise near borders as a proxy for damage
        h, w = edges.shape
        border_mask = np.zeros_like(edges, dtype=np.uint8)
        border_mask[0:5, :] = 1
        border_mask[h-5:h, :] = 1
        border_mask[:, 0:5] = 1
        border_mask[:, w-5:w] = 1
        border_edges = cv2.bitwise_and(edges, edges, mask=border_mask)
        # Normalize score: fewer edge pixels near borders => better condition
        max_val = border_mask.sum() * 255
        score = 1.0 - float(border_edges.sum()) / float(max_val)
        return round(max(0.0, min(1.0, score)), 2)
