"""
Enhanced Excel extractor (scaffold).
- Fuzzy column name mapping (OrderID vs Order Id vs PO Number).
"""

def extract_excel(path):
    # TODO: implement fuzzy header mapping
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
        "mode": "enhanced-excel"
    }
