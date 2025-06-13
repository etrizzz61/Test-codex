"""FastAPI application for card scanning."""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path

from .detection import CardDetector
from .ocr import CardOCR
from .data import CardDatabase
from .grading import CardConditionEvaluator

app = FastAPI(title="CardScanner API")

# Initialize components
MODEL_PATH = Path("models/yolov8_card.pt")
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "card_db.json"

detector = CardDetector(MODEL_PATH)
ocr_engine = CardOCR()
card_db = CardDatabase(DB_PATH)
grader = CardConditionEvaluator()


@app.post("/scan")
async def scan_card(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    bbox = detector.detect(tmp_path)
    ocr_result = ocr_engine.extract(tmp_path)
    card_info = {}
    if ocr_result.get("raw_text"):
        # naive search: look for card name in DB keys
        for name in card_db.data:
            if name.lower() in ocr_result["raw_text"].lower():
                card_info = card_db.find_card(name)
                break
    condition_score = grader.evaluate(tmp_path, bbox)
    return JSONResponse({
        "bbox": bbox,
        "ocr": ocr_result,
        "card_info": card_info,
        "condition_score": condition_score,
    })
