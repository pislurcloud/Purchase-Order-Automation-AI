from PIL import Image
import json

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def parse_key_values(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    data = {}
    items = []
    in_items = False
    for line in lines:
        if 'Items' in line:
            in_items = True
            continue
        if in_items:
            parts = line.split()
            if len(parts) >= 5:
                product_code = parts[0]
                qty = parts[-3]
                unit_price = parts[-2]
                total_price = parts[-1]
                description = ' '.join(parts[1:-3])
                items.append({
                    'product_code': product_code,
                    'description': description,
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
    try:
        import pytesseract
        img = Image.open(path).convert('RGB')
        text = pytesseract.image_to_string(img)
    except Exception as e:
        text = ''
    text = text.replace('\x0c','').strip()
    return parse_key_values(text)

if __name__ == '__main__':
    import sys, json
    res = extract_from_image(sys.argv[1])
    print(json.dumps(res, indent=2))
