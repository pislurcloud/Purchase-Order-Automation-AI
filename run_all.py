#!/usr/bin/env python3
"""
run_all.py
- Batch extractor for all files in data/mock_files
- Supports --mode baseline (default) or enhanced
"""

import argparse, json, traceback
from pathlib import Path

# paths
repo_root = Path(__file__).resolve().parent
mock_dir = repo_root / "data" / "mock_files"
out_dir = repo_root / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["baseline","enhanced"], default="baseline",
                    help="Extraction mode: baseline (stable) or enhanced (experimental)")
args = parser.parse_args()

mode = args.mode
print(f"[run_all] Running in {mode.upper()} mode")

# import handlers
if mode == "baseline":
    try:
        from src.extract_pdf import extract_from_pdf
    except ImportError:
        def extract_from_pdf(p): return {"error": "baseline PDF extractor missing"}
    try:
        from src.extract_excel import extract_excel
    except ImportError:
        def extract_excel(p): return {"error": "baseline Excel extractor missing"}
    try:
        from src.extract_ocr import extract_from_image
    except ImportError:
        def extract_from_image(p): return {"error": "baseline OCR extractor missing"}
else:
    try:
        from src.enhanced.extract_pdf_v2 import extract_from_pdf
    except ImportError:
        def extract_from_pdf(p): return {"error": "enhanced PDF extractor missing"}
    try:
        from src.enhanced.extract_excel_v2 import extract_excel
    except ImportError:
        def extract_excel(p): return {"error": "enhanced Excel extractor missing"}
    try:
        from src.enhanced.extract_ocr_v2 import extract_from_image
    except ImportError:
        def extract_from_image(p): return {"error": "enhanced OCR extractor missing"}

def extract_from_csv(path):
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return {
            "order_id": None,
            "client_name": None,
            "order_date": None,
            "delivery_date": None,
            "items": df.to_dict(orient="records"),
            "order_total": None,
            "currency": None,
            "special_instructions": None,
            "confidence_score": 0.5,
            "mode": mode
        }
    except Exception as e:
        return {"error": str(e)}

handlers = {
    ".pdf": lambda p: extract_from_pdf(str(p)),
    ".png": lambda p: extract_from_image(str(p)),
    ".jpg": lambda p: extract_from_image(str(p)),
    ".jpeg": lambda p: extract_from_image(str(p)),
    ".tiff": lambda p: extract_from_image(str(p)),
    ".bmp": lambda p: extract_from_image(str(p)),
    ".xlsx": lambda p: extract_excel(str(p)),
    ".xls": lambda p: extract_excel(str(p)),
    ".csv": lambda p: extract_from_csv(str(p)),
}

def main():
    processed = []
    for f in sorted(mock_dir.iterdir()):
        if not f.is_file() or f.name.startswith('.') or f.name.lower()=='.gitkeep':
            continue
        ext = f.suffix.lower()
        out_name = f.stem + f"_{mode}.json"
        out_path = out_dir / out_name
        try:
            handler = handlers.get(ext)
            if handler:
                res = handler(f)
            else:
                res = {"error": "unsupported file type: " + ext}
        except Exception as e:
            res = {"error": str(e), "traceback": traceback.format_exc()}
        try:
            with open(out_path, "w", encoding="utf-8") as fo:
                json.dump(res, fo, indent=2, default=str)
            processed.append(out_path)
            print("Wrote", out_path)
        except Exception as e:
            print("Failed writing", out_path, e)
    print("\nProcessed files:", [str(p) for p in processed])

if __name__ == "__main__":
    main()
