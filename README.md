# Purchase Order Extraction — PoC

This is an accelerated Proof of Concept (PoC) for purchase-order extraction.

It demonstrates:
- Client A: PDF table → JSON
- Client B: Excel multi-sheet → JSON
- Client E: Scanned form (OCR) → JSON
- Canonical JSON schema validation
- Basic Streamlit reviewer UI for human-in-the-loop corrections

## Run the PoC
```bash
pip install -r requirements.txt
python run_all.py
