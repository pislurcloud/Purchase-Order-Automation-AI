# src/extract_ocr.py  (improved preprocessing)
from PIL import Image
import pytesseract, json
import cv2
import numpy as np
from pathlib import Path

def ocr_preprocess_cv(path_or_pil):
    # Accept path string or PIL.Image
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert('RGB')
        img = np.array(img)
    else:
        img = np.array(path_or_pil.convert('RGB'))
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize to improve OCR (if small)
    h, w = gray.shape
    scale = 1.0
    if max(h,w) < 1000:
        scale = 2.0
    if scale != 1.0:
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    # Denoise
    gray = cv2.medianBlur(gray, 3)
    # Adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 9)
    # optionally invert if background dark
    # return PIL image for pytesseract
    return Image.fromarray(th)

def is_number(s):
    try:
        float(s); return True
    except: return False

def parse_key_values(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    data = {}
    items = []
    in_items = False
    for line in lines:
        if line.lower().startswith('items') or 'items' in line.lower():
            in_items = True
            continue
        if in_items:
            parts = line.split()
            if len(parts) >= 4:
                # flexible mapping: last two tokens are unit_price and total_price if numeric-like
                # find rightmost numeric tokens
                nums = [i for i,p in enumerate(parts) if (p.replace('.','',1).isdigit())]
                if len(nums) >= 2:
                    # assume qty is the token before the two prices if it's an int
                    total_price = parts[nums[-1]]
                    unit_price = parts[nums[-2]]
                    qty_idx = nums[-3] if len(nums)>=3 else nums[-2]-1
                    qty = parts[qty_idx] if qty_idx < len(parts) else ''
                    # description between code and qty_idx
                    product_code = parts[0]
                    desc = ' '.join(parts[1:qty_idx])
                    items.append({
                        'product_code': product_code,
                        'description': desc,
                        'quantity': int(qty) if qty.isdigit() else qty,
                        'unit_price': float(unit_price) if is_number(unit_price) else unit_price,
                        'total_price': float(total_price) if is_number(total_price) else total_price
                    })
                continue
        if ':' in line:
            k,v = line.split(':',1)
            data[k.strip().lower().replace(' ','_')] = v.strip()
    out = {
        'order_id': data.get('order_id'),
        'client_name': data.get('client_name'),
        'order_date': data.get('order_date'),
        'delivery_date': data.get('delivery_date'),
        'items': items,
        'order_total': sum([it.get('total_price',0) for it in items]) if items else None,
        'currency': data.get('currency'),
        'special_instructions': data.get('special_instructions'),
        'confidence_score': 0.75 if items else 0.5
    }
    return out

def extract_from_image(path):
    # Preprocess then OCR
    pre = ocr_preprocess_cv(path)
    text = pytesseract.image_to_string(pre)
    text = text.replace('\x0c','').strip()
    return parse_key_values(text)

if __name__ == '__main__':
    import sys; print(json.dumps(extract_from_image(sys.argv[1]), indent=2))
