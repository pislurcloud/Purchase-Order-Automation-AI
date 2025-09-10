"""
Enhanced PDF extractor (scaffold).
- OCR preprocessing hook (for scanned PDFs).
- Improved table parser placeholder.
"""

def extract_from_pdf(path):
    # TODO: implement OCR preprocessing + fuzzy table parsing
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
        "mode": "enhanced-pdf"
    }
