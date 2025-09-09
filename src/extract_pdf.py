from pathlib import Path

def extract_from_pdf(path):
    path = Path(path)
    try:
        import pdfplumber
        tables = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_tables()
                if t:
                    tables.extend(t)
        if tables:
            table = tables[0]
            headers = table[0]
            rows = table[1:]
            items = []
            for r in rows:
                d = {}
                for i,h in enumerate(headers):
                    key = (h or '').strip().lower().replace(' ','_')
                    d[key] = r[i] if i < len(r) else None
                items.append({
                    'product_code': d.get('product_code'),
                    'description': d.get('description'),
                    'quantity': int(d.get('quantity')) if d.get('quantity') else None,
                    'unit_price': float(d.get('unit_price')) if d.get('unit_price') else None,
                    'total_price': float(d.get('total_price')) if d.get('total_price') else None
                })
            return {
                'order_id': None,
                'client_name': None,
                'order_date': None,
                'delivery_date': None,
                'items': items,
                'order_total': sum([it.get('total_price',0) for it in items]),
                'currency': None,
                'special_instructions': None,
                'confidence_score': 0.96
            }
        else:
            texts = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    texts.append(page.extract_text() or '')
            combined = '\n'.join(texts)
            from src.extract_ocr import parse_key_values
            return parse_key_values(combined)
    except Exception as e:
        return {
            'order_id': None,
            'client_name': None,
            'order_date': None,
            'delivery_date': None,
            'items': [],
            'order_total': None,
            'currency': None,
            'special_instructions': None,
            'confidence_score': 0.0,
            'error': str(e)
        }

if __name__ == '__main__':
    import sys, json
    res = extract_from_pdf(sys.argv[1])
    print(json.dumps(res, indent=2))
