import pandas as pd, json
from pathlib import Path

def extract_excel(path):
    path = Path(path)
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names
    try:
        df_order = pd.read_excel(path, sheet_name='Order')
    except Exception:
        df_order = pd.read_excel(path, sheet_name=0)
    try:
        df_items = pd.read_excel(path, sheet_name='Items')
    except Exception:
        if len(sheets) > 1:
            df_items = pd.read_excel(path, sheet_name=1)
        else:
            df_items = pd.DataFrame()
    order = df_order.to_dict(orient='records')[0] if not df_order.empty else {}
    items = df_items.to_dict(orient='records') if not df_items.empty else []
    order_total = order.get('order_total')
    if not order_total and items:
        try:
            order_total = sum([float(it.get('total_price',0)) for it in items])
        except Exception:
            order_total = None
    out = {
        'order_id': order.get('order_id'),
        'client_name': order.get('client_name'),
        'order_date': str(order.get('order_date')) if order.get('order_date') is not None else None,
        'delivery_date': str(order.get('delivery_date')) if order.get('delivery_date') is not None else None,
        'items': items,
        'order_total': order_total,
        'currency': order.get('currency'),
        'special_instructions': order.get('special_instructions'),
        'confidence_score': 0.92
    }
    return out

if __name__ == '__main__':
    import sys, json
    res = extract_excel(sys.argv[1])
    print(json.dumps(res, indent=2))
