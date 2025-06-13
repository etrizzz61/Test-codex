# CardScanner

CardScanner is a prototype application that detects Pokémon and Lorcana cards from webcam or uploaded images. It uses YOLOv8 for object detection and EasyOCR for text extraction. A tiny JSON database provides official card data.

## Features

- Detect the card area in an image with YOLOv8.
- Extract printed text using EasyOCR.
- Lookup card information from a local JSON file.
- REST API built with FastAPI.
- Optional Gradio interface for quick testing.

## Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Running

Launch the API with Uvicorn:

```bash
uvicorn cardscanner.app.main:app --reload
```

Open the Gradio demo:

```bash
python -m cardscanner.app.gradio_app
```

## Notes

The YOLO model file `models/yolov8_card.pt` is expected to contain a trained model able to locate cards. Provide your own weights in that path.
