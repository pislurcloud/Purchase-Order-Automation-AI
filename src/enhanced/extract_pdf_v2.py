"""
Enhanced PDF extractor (v2).
- Uses pdfplumber for embedded text and tables; falls back to OCR with preprocessing.
- Improved table parsing: pdfplumber tables -> fuzzy text-table fallback.
- Metadata extraction (order id / dates / currency / totals).
- Client name extraction.
- Multiple line items supported.
- Diagnostics written into outputs/diagnostics/ for debugging.
"""

from pathlib import Path
import re, json
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import numpy as np
import cv2

DIAG_DIR = Path("outputs/diagnostics")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- OCR Preprocessing ---------------- #
def ocr_preprocess_cv(pil_image):
    img = np.array(pil_image.convert("RGB"))[:, :, ::-1].copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10,
                                        templateWindowSize=7, searchWindowSize=21)
    except Exception:
        pass
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    try:
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 12
        )
    except Exception:
        th = gray
    try:
        coords = np.column_stack(np.where(th > 0))
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = th.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        th = cv2.warpAffine(th, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass
    return Image.fromarray(th)

# ---------------- Metadata Parser ---------------- #
def parse_metadata(text):
    order_id = None
    order_date = delivery_date = currency = order_total = None

    od_patterns = [
        r"\bOrder\s*ID[:\s]*([A-Za-z0-9\-\_/]+)\b",
        r"\bPO[:\s\-]*([A-Za-z0-9\-\_/]+)\b",
        r"\bOrder\s*No[:\s]*([A-Za-z0-9\-\_/]+)\b",
        r"\bRef[:\s]*([A-Za-z0-9\-\_/]+)\b"
    ]
    for p in od_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            order_id = m.group(1).strip()
            break

    date_patterns = [
        r"(\d{4}[-/]\d{2}[-/]\d{2})",
        r"(\d{2}/\d{2}/\d{4})",
        r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
    ]
    found_dates = []
    for p in date_patterns:
        found_dates += re.findall(p, text)
    if found_dates:
        order_date = found_dates[0]
        if len(found_dates) > 1:
            delivery_date = found_dates[1]

    cur_m = re.search(r"\b(USD|EUR|GBP|INR|JPY|AUD|CAD)\b",
                      text, re.IGNORECASE)
    if cur_m:
        currency = cur_m.group(1).upper()
    else:
        if "€" in text:
            currency = "EUR"
        elif "£" in text:
            currency = "GBP"
        elif "$" in text:
            currency = "USD"

    total_m = re.search(
        r"(?:Order\s+Total|Grand Total|Total)[:\s]*([0-9\.,]+\d)",
        text, re.IGNORECASE)
    if total_m:
        try:
            order_total = float(total_m.group(1).replace(",", ""))
        except Exception:
            order_total = None
    else:
        nums = re.findall(r"([0-9\.,]+\d)", text)
        if nums:
            try:
                order_total = float(nums[-1].replace(",", ""))
            except Exception:
                order_total = None

    return order_id, order_date, delivery_date, currency, order_total

# ---------------- Client Name Parser ---------------- #
def parse_client_name(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:5]:
        low = ln.lower()
        if "purchase order" in low:
            if "-" in ln:
                return ln.split("-")[0].strip()
            return ln.split("purchase order")[0].strip()
    for ln in lines[:5]:
        if any(word in ln.lower() for word in
               ("corp","inc","ltd","llc","solutions","systems","technologies")):
            return ln.strip()
    return None

# ---------------- Table parsing helpers ---------------- #
def parse_tables_from_pdfplumber(tables):
    header_kws = ('product','description','qty','quantity','unit',
                  'price','total','code','item')
    results = []
    for table in tables:
        if not table or len(table) < 2:
            continue
        header_row = None
        header_idx = 0
        for i, row in enumerate(table[:3]):
            row_text = " ".join([str(x or "").lower() for x in row])
            if sum(1 for kw in header_kws if kw in row_text) >= 2:
                header_row = [str(x).strip() for x in row]
                header_idx = i
                break
        if header_row is None:
            header_row = [str(x).strip() for x in table[0]]
            header_idx = 0
        norm_headers = []
        for h in header_row:
            h_low = (h or "").lower()
            if 'product' in h_low or 'code' in h_low or 'item' in h_low:
                norm_headers.append('product_code')
            elif 'description' in h_low or 'desc' in h_low:
                norm_headers.append('description')
            elif 'qty' in h_low or 'quantity' in h_low:
                norm_headers.append('quantity')
            elif 'price' in h_low and 'total' not in h_low:
                norm_headers.append('unit_price')
            elif 'total' in h_low:
                norm_headers.append('total_price')
            else:
                norm_headers.append(h_low or 'col')
        for row in table[header_idx+1:]:
            if not any(cell for cell in row):
                continue
            row_map = dict(zip(norm_headers, [str(x).strip() if x else "" for x in row]))
            try:
                qty = int(float(row_map.get('quantity') or 0)) if row_map.get('quantity') else None
            except Exception:
                qty = None
            def to_float(s):
                try: return float(str(s).replace(",",""))
                except: return None
            item = {
                "product_code": row_map.get('product_code') or "",
                "description": row_map.get('description') or "",
                "quantity": qty,
                "unit_price": to_float(row_map.get('unit_price')),
                "total_price": to_float(row_map.get('total_price'))
            }
            if any([item["description"], item["product_code"], item["total_price"], item["unit_price"]]):
                results.append(item)
    return results

def parse_table_from_text(text):
    """
    Improved fallback parser.
    Handles multiple single-space separated rows like:
    PR-002 Gadget Max 31 46.34 1436.54
    PR-003 Widget Pro 12 55.10 661.20
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header_idx = None
    header_kws = ['product','description','qty','quantity','unit','price','total','code','item']
    for i, ln in enumerate(lines):
        if sum(1 for kw in header_kws if kw in ln.lower()) >= 2:
            header_idx = i
            break
    if header_idx is None:
        return []
    items = []
    for ln in lines[header_idx+1:]:
        low = ln.lower()
        if any(tok in low for tok in ('order id','order date','order total','currency','delivery','subtotal')):
            break
        parts = ln.strip().split()
        if len(parts) >= 5:
            product_code = parts[0]
            try:
                qty = int(parts[-3])
            except Exception:
                qty = None
            try:
                unit_price = float(parts[-2].replace(",", ""))
            except Exception:
                unit_price = None
            try:
                total_price = float(parts[-1].replace(",", ""))
            except Exception:
                total_price = None
            description = " ".join(parts[1:-3])
            items.append({
                "product_code": product_code,
                "description": description,
                "quantity": qty,
                "unit_price": unit_price,
                "total_price": total_price
            })
    return items

# ---------------- Diagnostics ---------------- #
def save_diagnostics(stem, raw_text, tables, meta):
    try:
        (DIAG_DIR / f"{stem}_raw.txt").write_text(raw_text[:20000], encoding="utf-8")
        if tables:
            (DIAG_DIR / f"{stem}_tables.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")
        (DIAG_DIR / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

# ---------------- Main Extractor ---------------- #
def extract_from_pdf(path):
    path = str(path)
    raw_text, extracted_tables = "", []
    stem = Path(path).stem

    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                raw_text += "\n" + (page.extract_text() or "")
                extracted_tables += page.extract_tables() or []
    except Exception:
        raw_text = ""

    if (not raw_text.strip()) or len(raw_text.strip()) < 50 or not extracted_tables:
        try:
            pages = convert_from_path(path, dpi=300, first_page=1, last_page=3)
            for img in pages:
                proc = ocr_preprocess_cv(img)
                raw_text += "\n" + pytesseract.image_to_string(proc)
        except Exception:
            pass

    order_id, order_date, delivery_date, currency, order_total = parse_metadata(raw_text)
    client_name = parse_client_name(raw_text)
    items = parse_tables_from_pdfplumber(extracted_tables) or parse_table_from_text(raw_text)

    found_count = sum(1 for v in (order_id, order_date, delivery_date, currency, order_total, client_name) if v)
    confidence = round(min(1.0, (found_count / 6.0) * 0.6 + (0.4 if items else 0)), 2)

    meta = {
        "order_id": order_id,
        "client_name": client_name,
        "order_date": order_date,
        "delivery_date": delivery_date,
        "currency": currency,
        "order_total": order_total,
        "item_count": len(items)
    }
    save_diagnostics(stem, raw_text, extracted_tables, meta)

    return {
        **meta,
        "items": items,
        "special_instructions": None,
        "confidence_score": confidence,
        "mode": "enhanced-pdf"
    }
