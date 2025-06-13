"""OCR module for extracting card text."""
from pathlib import Path
from typing import Dict, List

import easyocr
import cv2


class CardOCR:
    def __init__(self, languages: List[str] | None = None):
        langs = languages or ["en"]
        self.reader = easyocr.Reader(langs, gpu=False)

    def extract(self, image_path: str | Path) -> Dict[str, str]:
        """Extract text from the card image.

        Returns a dictionary with possible fields like name and set.
        """
        image = cv2.imread(str(image_path))
        result = self.reader.readtext(image, detail=0)
        # Basic heuristic: join all lines
        text = "\n".join(result)
        return {"raw_text": text}
