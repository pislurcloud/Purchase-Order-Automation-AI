"""
Enhanced OCR extractor (scaffold).
- OCR preprocessing (deskew, binarize, denoise).
"""

def extract_from_image(path):
    # TODO: implement OCR preprocessing + Tesseract call
    return {
        "order_id": None,
        "client_name": None,
        "order_date": None,
        "delivery_date": None,
        "items": [],
        "order_total": None,
        "currency": None,
        "special_instructions": None,
        "confidence_score": 0.0,
        "mode": "enhanced-ocr"
    }
