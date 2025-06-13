"""Gradio interface for CardScanner."""
import gradio as gr
from pathlib import Path
from .main import detector, ocr_engine, card_db, grader


def process_image(image: str):
    path = Path(image)
    bbox = detector.detect(path)
    ocr_result = ocr_engine.extract(path)
    card_info = {}
    if ocr_result.get("raw_text"):
        for name in card_db.data:
            if name.lower() in ocr_result["raw_text"].lower():
                card_info = card_db.find_card(name)
                break
    condition = grader.evaluate(path, bbox)
    return {
        "bbox": bbox,
        "ocr_text": ocr_result.get("raw_text"),
        "card_info": card_info,
        "condition_score": condition,
    }


def main():
    demo = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="filepath"),
        outputs=["json"],
        title="CardScanner",
        description="Upload an image of a Pokémon or Lorcana card to identify it.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
