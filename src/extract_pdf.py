# src/extract_pdf.py
# Robust PDF extractor: tries pdfplumber tables + page text for metadata,
# falls back to rendering PDF pages to images and doing OCR, and uses
# tightened metadata regexes with header-line pruning and text-table parsing.

from pathlib import Path
import re
import json

# --- Improved metadata parsing helpers ---
RE_ORDER_ID_EXPLICIT = re.compile(
    r'(?:order[_\s\-]*id|po[_\s\-]*no|po[:\s\-])\s*[:#\-]?\s*([A-Z0-9\-\_]+)', re.I
)
RE_PO_DASH = re.compile(r'\b(PO[-\s]?[A-Z0-9\-_]+)\b', re.I)          # PO-123, PO123, PO-ABC
RE_ORDER_ID_GENERIC = re.compile(r'\b([A-Z]{2,5}[-\d]{2,20})\b')     # fallback (careful)

RE_DATE = re.compile(r'(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})')
RE_TOTAL = re.compile(
    r'(?:order[_\s\-]*total|grand[_\s\-]*total|total[:\s])[:\s]*([0-9,]+(?:\.\d{1,2})?)',
    re.I
)
CURRENCY_KEYWORDS = ['USD','INR','EUR','GBP','AUD','CAD','JPY','CNY']

HEADER_BLACKLIST = ['product','description','quantity','unit price','total price','items','notes','order total']


def _remove_header_lines(text):
    """Remove lines that look like table headers to avoid false-positive regex matches."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    filtered = []
    for ln in lines:
        low = ln.lower()
        hits = sum(1 for h in HEADER_BLACKLIST if h in low)
        if hits >= 1 and len(low.split()) <= 10:
            continue
        filtered.append(ln)
    return "\n".join(filtered)


def first_regex(pattern, text):
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def normalize_number(s):
    if s is None:
        return None
    s = str(s).replace(',', '').strip()
    try:
        return float(s)
    except:
        return None


def parse_table_from_text(text):
    """
    Look for a header line containing product / description / quantity / price tokens,
    then parse subsequent lines as rows until a blank or non-row line is seen.
    Returns list of item dicts.
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # find header index
    header_idx = None
    header_keywords = ['product', 'product code', 'description', 'quantity', 'qty', 'unit price', 'unit_price', 'total price', 'amount']
    for i, ln in enumerate(lines):
        low = ln.lower()
        hits = sum(1 for k in header_keywords if k in low)
        if hits >= 2:
            header_idx = i
            break
    if header_idx is None:
        return []

    header_line = lines[header_idx]
    import re as _re
    cols = _re.split(r'\s{2,}|\t', header_line)
    if len(cols) <= 1:
        cols = header_line.split()

    items = []
    # rows are the subsequent lines until a line that looks like metadata (contains ':' or 'order' 'total' etc.)
    for ln in lines[header_idx+1:]:
        low = ln.lower()
        if ':' in ln and ('order' in low or 'date' in low or 'total' in low or 'currency' in low):
            break
        parts = _re.split(r'\s{2,}|\t', ln)
        if len(parts) <= 1:
            parts = ln.split()
        if len(parts) >= 4:
            try:
                code = parts[0]
                # default mapping: last 3 tokens likely qty, unit_price, total_price
                if len(parts) >= 5:
                    qty = parts[-3]
                    unit = parts[-2]
                    total = parts[-1]
                    desc = ' '.join(parts[1:-3])
                elif len(parts) == 4:
                    # try: code desc qty total
                    qty = parts[-2]
                    total = parts[-1]
                    unit = None
                    desc = parts[1]
                else:
                    qty = None; unit = None; total = None
                    desc = ' '.join(parts[1:])
                def tonum(x):
                    try:
                        return float(str(x).replace(',',''))
                    except:
                        return None
                quantity_val = None
                if qty is not None:
                    try:
                        quantity_val = int(qty)
                    except:
                        try:
                            quantity_val = int(float(qty))
                        except:
                            quantity_val = qty
                items.append({
                    'product_code': code,
                    'description': desc,
                    'quantity': quantity_val,
                    'unit_price': tonum(unit),
                    'total_price': tonum(total)
                })
            except Exception:
                continue
        else:
            continue

    return items


def parse_metadata_from_text(text):
    """
    Robust metadata parser:
    - prunes table header lines to avoid false positives (e.g., 'Product' contains 'PO')
    - prefers explicit 'Order ID' and PO- patterns
    - extracts dates (prefers ISO yyyy-mm-dd)
    - finds currency and totals heuristically
    """
    text = text or ""
    pruned_text = _remove_header_lines(text)

    md = {
        'order_id': None,
        'order_date': None,
        'delivery_date': None,
        'currency': None,
        'order_total': None,
        'client_name': None,
        'special_instructions': None
    }

    # 1) explicit 'Order ID' / 'PO No' patterns
    order_explicit = first_regex(RE_ORDER_ID_EXPLICIT, pruned_text)
    if order_explicit:
        md['order_id'] = order_explicit

    # 2) look for PO-... pattern (word boundary)
    if not md['order_id']:
        po = first_regex(RE_PO_DASH, pruned_text)
        if po:
            md['order_id'] = po.replace(' ', '').upper()

    # 3) search lines containing 'order' or 'po' for PO- tokens (safer)
    if not md['order_id']:
        lines = [ln for ln in pruned_text.splitlines() if 'order' in ln.lower() or 'po' in ln.lower()]
        for ln in lines:
            m = RE_PO_DASH.search(ln)
            if m:
                md['order_id'] = m.group(1).replace(' ', '').upper()
                break
        # fallback generic (only if it clearly starts with PO)
        if not md['order_id']:
            gen = first_regex(RE_ORDER_ID_GENERIC, pruned_text)
            if gen and gen.upper().startswith('PO'):
                md['order_id'] = gen

    # Dates: robust splitting even when on same line
    dates = RE_DATE.findall(pruned_text)
    if dates:
        iso = [d for d in dates if re.match(r'\d{4}[-/]\d{2}[-/]\d{2}', d)]
        if iso:
            md['order_date'] = iso[0]
            # try to find a second date after the first occurrence in text
            first_pos = None
            for m in RE_DATE.finditer(pruned_text):
                if m.group(0) == iso[0]:
                    first_pos = m.end()
                    break
            if first_pos:
                next_m = RE_DATE.search(pruned_text, pos=first_pos)
                if next_m:
                    md['delivery_date'] = next_m.group(0)
        else:
            md['order_date'] = dates[0]
            if len(dates) > 1:
                md['delivery_date'] = dates[1]

    # Currency: explicit 'Currency:' first, else common codes in text
    cur_search = re.search(r'currency[:\s]*([A-Z]{3})', pruned_text, re.I)
    if cur_search:
        md['currency'] = cur_search.group(1).upper()
    else:
        for c in CURRENCY_KEYWORDS:
            if c in pruned_text:
                md['currency'] = c
                break

    # Order total
    total = first_regex(RE_TOTAL, pruned_text)
    if total:
        md['order_total'] = normalize_number(total)
    else:
        # fallback: look for numeric token on lines containing 'total'
        for ln in pruned_text.splitlines():
            if 'total' in ln.lower():
                nums = re.findall(r'([0-9\.,]+)', ln)
                if nums:
                    md['order_total'] = normalize_number(nums[-1])
                    break

    # client_name heuristic: look for a line before the 'Order' or title but not the generic "Purchase Order" header
    lines = [ln.strip() for ln in pruned_text.splitlines() if ln.strip()]
    client = None
    for i, ln in enumerate(lines[:8]):  # first few lines
        low = ln.lower()
        if 'purchase order' in low or 'invoice' in low:
            # try previous line if exists
            if i > 0:
                candidate = lines[i-1].strip()
                if len(candidate) > 2:
                    client = candidate
                    break
            continue
        if any(h in low for h in HEADER_BLACKLIST):
            continue
        if len(ln) > 2:
            client = ln
            break
    md['client_name'] = client or md.get('client_name')

    return md


# --- End metadata helpers ---


def extract_from_pdf(path):
    """
    Main extractor: tries pdfplumber for tables and page text,
    then falls back to pdf2image + OCR (using src.extract_ocr.parse_key_values).
    Returns a dict matching the canonical PoC schema (not strict JSON Schema).
    """
    path = Path(path)
    extracted_items = []
    metadata = {
        'order_id': None,
        'client_name': None,
        'order_date': None,
        'delivery_date': None,
        'order_total': None,
        'currency': None,
        'special_instructions': None
    }

    # Try pdfplumber for structured tables and page text
    combined_page_text = ""
    try:
        import pdfplumber
        page_text_accum = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ''
                page_text_accum.append(txt)
                t = page.extract_tables()
                if t:
                    table = t[0]
                    if len(table) >= 2:
                        headers = [ (h or '').strip().lower().replace(' ','_') for h in table[0] ]
                        rows = table[1:]
                        for r in rows:
                            rowd = {}
                            for i, h in enumerate(headers):
                                rowd[h] = r[i] if i < len(r) else None
                            product_code = rowd.get('product_code') or rowd.get('product') or rowd.get('sku')
                            description = rowd.get('description') or rowd.get('item') or ''
                            qty = rowd.get('quantity') or rowd.get('qty')
                            unit_price = rowd.get('unit_price') or rowd.get('price')
                            total_price = rowd.get('total_price') or rowd.get('amount')
                            try:
                                qtyv = int(str(qty)) if qty not in (None, '') else None
                            except:
                                qtyv = None
                            try:
                                upv = float(str(unit_price).replace(',', '')) if unit_price not in (None, '') else None
                            except:
                                upv = None
                            try:
                                tpv = float(str(total_price).replace(',', '')) if total_price not in (None, '') else None
                            except:
                                tpv = None
                            extracted_items.append({
                                'product_code': product_code,
                                'description': description,
                                'quantity': qtyv,
                                'unit_price': upv,
                                'total_price': tpv
                            })
        combined_page_text = "\n".join([t for t in page_text_accum if t])
        # parse metadata from combined text
        md = parse_metadata_from_text(combined_page_text)
        metadata.update(md)
    except Exception:
        # pdfplumber failure is non-fatal; we'll try OCR fallback below
        combined_page_text = ""

    # If no items found via tables, attempt to parse table-like text from page text
    if not extracted_items and combined_page_text:
        parsed_items = parse_table_from_text(combined_page_text)
        if parsed_items:
            extracted_items = parsed_items

    # If still no items found via tables, fallback to PDF->images -> OCR
    if not extracted_items:
        try:
            from pdf2image import convert_from_path
            from src.extract_ocr import ocr_preprocess_cv, parse_key_values
            import pytesseract
            imgs = convert_from_path(str(path), dpi=300)
            all_texts = []
            for img in imgs:
                pre = ocr_preprocess_cv(img)
                t = pytesseract.image_to_string(pre)
                all_texts.append(t)
            combined_ocr = "\n".join(all_texts)

            # Try to parse table from OCR text first
            parsed_items = parse_table_from_text(combined_ocr)
            if parsed_items:
                extracted_items = parsed_items
            else:
                parsed = parse_key_values(combined_ocr)
                if isinstance(parsed, dict):
                    extracted_items = parsed.get('items', []) or []
                    # merge parsed metadata
                    for k in ['order_id', 'order_date', 'delivery_date', 'currency', 'special_instructions', 'order_total', 'client_name']:
                        if parsed.get(k):
                            metadata[k] = parsed.get(k)

            # If some metadata still missing, try regex on the combined OCR text and page text
            if not any(metadata.values()):
                metadata.update(parse_metadata_from_text(combined_ocr))
            if combined_page_text:
                # prefer page text metadata when present
                metadata.update({k: v for k, v in parse_metadata_from_text(combined_page_text).items() if v})
        except Exception as e:
            return {
                'order_id': metadata.get('order_id'),
                'client_name': metadata.get('client_name'),
                'order_date': metadata.get('order_date'),
                'delivery_date': metadata.get('delivery_date'),
                'items': extracted_items,
                'order_total': metadata.get('order_total'),
                'currency': metadata.get('currency'),
                'special_instructions': metadata.get('special_instructions'),
                'confidence_score': 0.0,
                'error': str(e)
            }

    # compute order_total if missing from metadata
    if not metadata.get('order_total'):
        try:
            total = sum([it.get('total_price') or 0 for it in extracted_items])
            metadata['order_total'] = total if total != 0 else metadata.get('order_total')
        except:
            pass

    # Improve dates: if order_date still contains both dates (rare), split using regex
    if metadata.get('order_date') and 'delivery' in str(metadata.get('order_date')).lower():
        # attempt to split two dates in the string
        found = RE_DATE.findall(str(metadata.get('order_date')))
        if found:
            metadata['order_date'] = found[0]
            if len(found) > 1:
                metadata['delivery_date'] = found[1]

    result = {
        'order_id': metadata.get('order_id'),
        'client_name': metadata.get('client_name'),
        'order_date': metadata.get('order_date'),
        'delivery_date': metadata.get('delivery_date'),
        'items': extracted_items,
        'order_total': metadata.get('order_total'),
        'currency': metadata.get('currency'),
        'special_instructions': metadata.get('special_instructions'),
        'confidence_score': 0.9 if extracted_items else 0.6
    }
    return result


if __name__ == '__main__':
    import sys
    print(json.dumps(extract_from_pdf(sys.argv[1]), indent=2))
