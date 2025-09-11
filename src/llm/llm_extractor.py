"""
LLM extractor orchestrator (experimental)
- Accepts text/table inputs and routes to adapters (OpenAI/HF/local).
- Returns canonical JSON matching project schema.
- This is a scaffold for development — adapters are pluggable.
"""

from pathlib import Path
import json

SCHEMA = {
    "order_id": None,
    "client_name": None,
    "order_date": None,
    "delivery_date": None,
    "items": [],
    "order_total": None,
    "currency": None,
    "special_instructions": None,
    "confidence_score": 0.0
}

def extract_with_llm(raw_text, table_csv=None, context=None, provider="openai"):
    """
    raw_text: OCR or extracted text
    table_csv: optional small CSV string with table rows
    context: dict with metadata (filename, client hints)
    provider: 'openai' | 'hf' | 'local' - adapter selected
    Returns: dict conforming to SCHEMA
    """
    # This is a minimal stub. Integration with adapters goes here.
    result = dict(SCHEMA)
    result["special_instructions"] = None
    # For now, echo a best-effort placeholder
    result["confidence_score"] = 0.0
    result["_llm_meta"] = {"provider": provider, "note": "stub - implement adapter calls"}
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/llm/llm_extractor.py path/to/diagnostic_raw.txt")
        sys.exit(1)
    p = Path(sys.argv[1])
    txt = p.read_text(encoding="utf-8")
    out = extract_with_llm(txt)
    print(json.dumps(out, indent=2))
