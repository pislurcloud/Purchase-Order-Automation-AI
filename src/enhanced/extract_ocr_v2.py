"""
Enhanced OCR extractor for images (v2) - patched for robust OCR postprocessing

Highlights:
- Upscales image before OCR (helps small fonts).
- Tries multiple PSM modes (3,4,6,11,12) and picks best output by digit count heuristic.
- Normalizes OCR text (split CamelCase, fix '47x' -> '47 x', etc).
- Multiple regex patterns for items (with/without codes, with 'x' markers).
- Writes diagnostics to outputs/diagnostics/<stem>_raw.txt and <stem>_meta.json
"""

from pathlib import Path
import re, json
from PIL import Image
import pytesseract
import numpy as np
import cv2

DIAG_DIR = Path("outputs/diagnostics")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- OCR preprocessing ----------------
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

# ---------------- OCR variants ----------------
def run_tesseract_variants(pil_img):
    """Run Tesseract with multiple PSM modes and return best text by heuristic (most digits)."""
    configs = [
        "--oem 1 --psm 3",  # default, fully automatic page segmentation
        "--oem 1 --psm 4",  # assume columns
        "--oem 1 --psm 6",  # assume uniform block of text
        "--oem 1 --psm 11", # sparse text
        "--oem 1 --psm 12"  # sparse text with OSD
    ]
    best_text = ""
    best_score = -1
    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(pil_img, config=cfg)
            score = len(re.findall(r"\d", txt))
            if score > best_score:
                best_text = txt
                best_score = score
        except Exception:
            continue
    return best_text

# ---------------- Normalization ----------------
def normalize_ocr_text(text):
    if not text:
        return ""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]+', '', text)
    text = re.sub(r'^[\W_]+', '', text, flags=re.M)
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = text.replace('×', 'x')
    text = re.sub(r'(\d)([xX])(?=\S)', r'\1 x ', text)
    text = re.sub(r'([xX])(\d)', r' x \2', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = '\n'.join([ln.strip() for ln in text.splitlines() if ln.strip()])
    return text

# ---------------- Metadata parsing ----------------
def parse_metadata(text):
    order_id = None
    order_date = delivery_date = currency = order_total = None

    od_patterns = [
        r"\b(?:Order\s*ID|Order\s*No|Ref)[:\s]*([A-Za-z0-9\-\_/]+)\b",
        r"\b([A-Z0-9]{1,3}-\d{2,6})\b",
        r"\bPO[:\s\-]*([A-Za-z0-9\-\_/]+)\b"
    ]
    for p in od_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            order_id = re.sub(r'^[^A-Za-z0-9]+', '', m.group(1).strip())
            break

    date_patterns = [
        r"(\d{4}[-/]\d{2}[-/]\d{2})",
        r"(\d{2}/\d{2}/\d{4})",
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
    ]
    found_dates = []
    for p in date_patterns:
        found_dates += re.findall(p, text)
    if found_dates:
        def normalize_date_str(s):
            s = s.strip()
            m = re.match(r'(\d{1,2})[-/](\d{1,2})[-/](\d{2})$', s)
            if m:
                dd, mm, yy = m.groups()
                return f"20{yy}-{int(mm):02d}-{int(dd):02d}"
            m2 = re.match(r'(\d{4})[-/](\d{2})[-/](\d{2})$', s)
            if m2:
                return s
            m3 = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})$', s)
            if m3:
                dd, mm, yyyy = m3.groups()
                return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
            return s
        order_date = normalize_date_str(found_dates[0])
        if len(found_dates) > 1:
            delivery_date = normalize_date_str(found_dates[1])

    cur_m = re.search(r"\b(USD|EUR|GBP|INR|JPY|AUD|CAD)\b", text, re.IGNORECASE)
    if cur_m:
        currency = cur_m.group(1).upper()
    else:
        if "€" in text: currency = "EUR"
        elif "£" in text: currency = "GBP"
        elif "$" in text: currency = "USD"

    total_m = re.search(r"(?:Order\s+Total|Grand Total|Total)[:\s]*([0-9\.,]+\d)", text, re.IGNORECASE)
    if total_m:
        try: order_total = float(total_m.group(1).replace(",", ""))
        except: order_total = None
    else:
        nums = re.findall(r"([0-9\.,]+\d)", text)
        if nums:
            try: order_total = float(nums[-1].replace(",", ""))
            except: order_total = None

    return order_id, order_date, delivery_date, currency, order_total

# ---------------- Client name ----------------
def parse_client_name(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:6]:
        if "purchase order" in ln.lower():
            if "-" in ln:
                return ln.split("-")[0].strip()
            return ln.split("purchase order")[0].strip()
    for ln in lines[:6]:
        if any(w in ln.lower() for w in ("corp","inc","ltd","llc","solutions","systems","designs","technologies")):
            return ln.strip()
    return lines[0].strip() if lines else None

# ---------------- Item parsing ----------------
_item_patterns = [
    re.compile(r'^\s*(?P<code>[A-Za-z0-9\-\_]+)\s+(?P<desc>.+?)\s+(?P<qty>\d+)\s*x\s*(?P<unit>[\d,]+\.\d{1,2})(?:\s+(?P<total>[\d,]+\.\d{1,2}))?\s*$', re.IGNORECASE),
    re.compile(r'^\s*\d+\s+(?P<code>[A-Za-z0-9\-\_]+)\s+(?P<desc>.+?)\s+(?P<qty>\d+)\s*x\s*(?P<unit>[\d,]+\.\d{1,2})\s*$', re.IGNORECASE),
    re.compile(r'^\s*(?P<desc>[A-Za-z0-9\-\s]+?)\s+(?P<qty>\d+)\s*x\s*(?P<unit>[\d,]+\.\d{1,2})\s*$', re.IGNORECASE),
    re.compile(r'^\s*(?P<code>[A-Za-z0-9\-\_]+)?\s*(?P<desc>.*?)\s+(?P<qty>\d+)\s+(?P<unit>[\d,]+\.\d{1,2})\s+(?P<total>[\d,]+\.\d{1,2})\s*$', re.IGNORECASE),
]

def try_parse_line_for_item(line):
    ln = line.strip()
    for pat in _item_patterns:
        m = pat.match(ln)
        if m:
            d = m.groupdict()
            code = d.get('code') or ""
            desc = d.get('desc') or ""
            qty = None
            unit_price = None
            total_price = None
            if d.get('qty'):
                try: qty = int(d['qty'].replace(',', ''))
                except: qty = None
            if d.get('unit'):
                try: unit_price = float(d['unit'].replace(',', ''))
                except: unit_price = None
            if d.get('total'):
                try: total_price = float(d['total'].replace(',', ''))
                except: total_price = None
            if total_price is None and qty and unit_price:
                total_price = round(qty * unit_price, 2)
            desc = re.sub(r'\s{2,}', ' ', desc).strip()
            if code and re.fullmatch(r'\d+', code) and desc:
                parts = desc.split()
                if parts:
                    potential_code = parts[0]
                    if re.match(r'^[A-Za-z0-9\-\_]+$', potential_code):
                        desc = ' '.join(parts[1:]) if len(parts) > 1 else ''
                        code = potential_code
            return {
                "product_code": code or None,
                "description": desc or None,
                "quantity": qty,
                "unit_price": unit_price,
                "total_price": total_price
            }
    return None

# ---------------- Main extractor ----------------
def save_diagnostics(stem, raw_text, meta):
    try:
        (DIAG_DIR / f"{stem}_raw.txt").write_text(raw_text[:20000], encoding="utf-8")
        (DIAG_DIR / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

def extract_from_image(path):
    path = str(path)
    stem = Path(path).stem
    raw_text = ""
    try:
        img = Image.open(path)
        proc = ocr_preprocess_cv(img)
        cv_img = np.array(proc.convert("RGB"))[:, :, ::-1].copy()
        cv_img = cv2.resize(cv_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        proc = Image.fromarray(cv_img[:, :, ::-1])
        raw_text = run_tesseract_variants(proc)
        if not raw_text.strip():
            raw_text = run_tesseract_variants(img)
    except Exception as e:
        return {"error": f"ocr failed: {e}", "mode": "enhanced-ocr"}

    normalized = normalize_ocr_text(raw_text)
    lines = [ln for ln in normalized.splitlines() if ln.strip()]
    merged_lines = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if try_parse_line_for_item(cur) is None and i+1 < len(lines):
            joined = cur + " " + lines[i+1]
            if try_parse_line_for_item(joined):
                merged_lines.append(joined)
                i += 2
                continue
        merged_lines.append(cur)
        i += 1

    items = [itm for ln in merged_lines if (itm := try_parse_line_for_item(ln))]

    order_id, order_date, delivery_date, currency, order_total = parse_metadata(normalized)
    client_name = parse_client_name(normalized)

    meta_present = sum(1 for v in (order_id, order_date, delivery_date, currency, order_total, client_name) if v)
    confidence = round(min(1.0, (meta_present / 6.0) * 0.6 + (0.4 if items else 0)), 2)

    meta = {
        "order_id": order_id,
        "client_name": client_name,
        "order_date": order_date,
        "delivery_date": delivery_date,
        "currency": currency,
        "order_total": order_total,
        "item_count": len(items)
    }
    save_diagnostics(stem, normalized, meta)

    return {
        **meta,
        "items": items,
        "special_instructions": None,
        "confidence_score": confidence,
        "mode": "enhanced-ocr"
    }
