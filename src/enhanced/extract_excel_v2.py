"""
Enhanced Excel extractor (v2)
- Supports multi-sheet workbooks (Orders, Items).
- Fuzzy header mapping to canonical fields (order_id, order_date, client_name, qty, unit_price, total_price).
- Will try to extract metadata from a metadata sheet or top-left cells if present.
- Writes diagnostics to outputs/diagnostics/<stem>_meta.json and raw CSV previews.
"""

from pathlib import Path
import json, re
import pandas as pd

DIAG_DIR = Path("outputs/diagnostics")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# canonical header mapping by normalized token
CANONICAL_MAP = {
    # order-level
    'orderid': 'order_id', 'order_id': 'order_id', 'orderno': 'order_id', 'po': 'order_id', 'ponumber': 'order_id', 'po_no': 'order_id',
    'orderdate': 'order_date', 'po_date': 'order_date', 'date': 'order_date',
    'deliverydate': 'delivery_date', 'shipdate': 'delivery_date',
    'customer': 'client_name', 'client': 'client_name', 'company': 'client_name',
    # item-level
    'product': 'product_code', 'productcode': 'product_code', 'code': 'product_code', 'itemcode': 'product_code',
    'description': 'description', 'desc': 'description',
    'qty': 'quantity', 'quantity': 'quantity', 'amount': 'total_price',
    'unitprice': 'unit_price', 'price': 'unit_price', 'total': 'total_price', 'line_total': 'total_price'
}

# utility helpers
def norm_colname(c):
    if c is None:
        return ""
    return re.sub(r'[^a-z0-9]', '', str(c).strip().lower())

def map_columns(cols):
    mapped = {}
    for c in cols:
        nc = norm_colname(c)
        if nc in CANONICAL_MAP:
            mapped[c] = CANONICAL_MAP[nc]
        else:
            # fuzzy tries
            if 'order' in nc and 'id' in nc:
                mapped[c] = 'order_id'
            elif 'date' in nc and ('order' in nc or 'po' in nc):
                mapped[c] = 'order_date'
            elif 'cust' in nc or 'client' in nc or 'company' in nc:
                mapped[c] = 'client_name'
            elif 'qty' in nc or 'quantity' in nc:
                mapped[c] = 'quantity'
            elif 'unit' in nc and 'price' in nc or 'unitprice' in nc:
                mapped[c] = 'unit_price'
            elif 'price' in nc or 'total' in nc or 'amount' in nc:
                mapped[c] = 'total_price'
            elif 'desc' in nc:
                mapped[c] = 'description'
            elif 'prod' in nc or 'code' in nc:
                mapped[c] = 'product_code'
            else:
                mapped[c] = None
    return mapped

def extract_metadata_from_sheet(df):
    # scan top-left region for order metadata (first 10 rows/cols)
    meta = {}
    try:
        # check some common metadata column names first
        for c in df.columns[:6]:
            nc = norm_colname(c)
            if nc in ('orderid','ponumber','orderno','po'):
                meta['order_id'] = str(df[c].dropna().astype(str).iloc[0])
            if 'date' in nc and 'order' in nc or nc in ('orderdate','date','po_date'):
                meta['order_date'] = str(df[c].dropna().astype(str).iloc[0])
            if 'cust' in nc or 'client' in nc or 'company' in nc:
                meta['client_name'] = str(df[c].dropna().astype(str).iloc[0])
    except Exception:
        pass
    # fallback: find key/value pairs in first few rows
    try:
        head = df.head(10)
        for _, row in head.iterrows():
            for k, v in row.items():
                if str(k).strip().lower() in ('order id','orderid','po','po number','po_no','orderno'):
                    meta.setdefault('order_id', str(v))
    except Exception:
        pass
    return meta

def to_float_maybe(v):
    try:
        if pd.isna(v):
            return None
        s = str(v).replace(',','').strip()
        if s=='':
            return None
        return float(s)
    except Exception:
        return None

def canonical_output(order_id=None, client_name=None, order_date=None, delivery_date=None, items=None, order_total=None, currency=None, confidence=0.0, mode="enhanced-excel"):
    return {
        "order_id": order_id,
        "client_name": client_name,
        "order_date": order_date,
        "delivery_date": delivery_date,
        "items": items or [],
        "order_total": order_total,
        "currency": currency,
        "special_instructions": None,
        "confidence_score": confidence,
        "mode": mode
    }

def save_diag(stem, meta, preview=None):
    try:
        (DIAG_DIR / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if preview is not None:
            (DIAG_DIR / f"{stem}_preview.csv").write_text(preview, encoding="utf-8")
    except Exception:
        pass

def extract_excel(path):
    path = str(path)
    stem = Path(path).stem
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        return {"error": f"failed reading excel: {e}", "mode": "enhanced-excel"}

    sheets = xls.sheet_names
    # try to find Items sheet
    items_df = None
    orders_meta = {}
    # Common sheet names to try first
    for name in sheets:
        ln = name.lower()
        if ln in ('items','lines','orderlines','lineitems','order_items'):
            try:
                items_df = pd.read_excel(path, sheet_name=name)
                break
            except Exception:
                items_df = None
    # if not found, heuristically find sheet that has qty/product columns
    if items_df is None:
        for name in sheets:
            try:
                df = pd.read_excel(path, sheet_name=name)
                cols = [norm_colname(c) for c in df.columns]
                if any(c for c in cols if 'qty' in c or 'quantity' in c) and any(c for c in cols if 'prod' in c or 'code' in c or 'description' in c):
                    items_df = df
                    break
            except Exception:
                continue

    # Extract metadata from a "Orders" or first sheet
    for candidate in ('orders','order','metadata','summary'):
        if candidate in [s.lower() for s in sheets]:
            try:
                md_df = pd.read_excel(path, sheet_name=candidate)
                orders_meta.update(extract_metadata_from_sheet(md_df))
            except Exception:
                pass

    if not orders_meta:
        # fallback: read first sheet top rows
        try:
            first = pd.read_excel(path, sheet_name=0)
            orders_meta.update(extract_metadata_from_sheet(first))
        except Exception:
            pass

    # If items_df still None, pick second sheet if exists
    if items_df is None and len(sheets) >= 2:
        try:
            items_df = pd.read_excel(path, sheet_name=1)
        except Exception:
            items_df = None

    items = []
    mapped = {}
    order_total = None
    currency = None

    if items_df is not None:
        # preview CSV
        try:
            preview_csv = items_df.head(50).to_csv(index=False)
        except Exception:
            preview_csv = ""
        # map columns
        col_map = map_columns(items_df.columns)
        # try to find best columns by mapped canonical names
        cols_used = {}
        for orig_col, canonical in col_map.items():
            if canonical:
                cols_used[canonical] = orig_col
        # produce items rows
        for _, row in items_df.iterrows():
            # prefer keys: product_code, description, quantity, unit_price, total_price
            product_code = row.get(cols_used.get('product_code')) if cols_used.get('product_code') else None
            description = row.get(cols_used.get('description')) if cols_used.get('description') else None
            quantity = row.get(cols_used.get('quantity')) if cols_used.get('quantity') else None
            unit_price = row.get(cols_used.get('unit_price')) if cols_used.get('unit_price') else None
            total_price = row.get(cols_used.get('total_price')) if cols_used.get('total_price') else None
            # coerce numeric
            try:
                qty = int(float(quantity)) if quantity is not None and str(quantity).strip()!='' else None
            except:
                qty = None
            up = to_float_maybe(unit_price)
            tp = to_float_maybe(total_price)
            item = {
                "product_code": str(product_code).strip() if product_code is not None else None,
                "description": str(description).strip() if description is not None else None,
                "quantity": qty,
                "unit_price": up,
                "total_price": tp
            }
            # only add if there is something meaningful
            if any([item['product_code'], item['description'], item['total_price'], item['unit_price']]):
                items.append(item)
        # attempt to find order total in items sheet trailing rows
        try:
            # if there is a 'Total' column, maybe summary row
            if 'total_price' in cols_used and not order_total:
                # sum of item totals if present
                s = sum([r['total_price'] for r in items if r['total_price'] is not None])
                if s:
                    order_total = float(s)
        except Exception:
            pass
    else:
        preview_csv = ""

    # fill metadata from orders_meta if present
    order_id = orders_meta.get('order_id')
    client_name = orders_meta.get('client_name')
    order_date = orders_meta.get('order_date')
    delivery_date = orders_meta.get('delivery_date')

    # If still missing order_id and there's a single item, try from items_df column that looks like order id
    if not order_id:
        # search entire workbook first 5 sheets for a key 'Order ID' in first column
        try:
            for name in sheets[:3]:
                try:
                    df = pd.read_excel(path, sheet_name=name, header=None)
                    for i in range(min(10, len(df))):
                        row = df.iloc[i]
                        for j,cell in enumerate(row.fillna('').astype(str)):
                            if 'order id' in str(cell).lower() or 'order number' in str(cell).lower() or 'po'==str(cell).strip().lower():
                                # read the cell next to it
                                val = df.iloc[i, j+1] if j+1 < df.shape[1] else None
                                if val and str(val).strip():
                                    order_id = str(val).strip()
                                    break
                        if order_id:
                            break
                except Exception:
                    pass
                if order_id:
                    break
        except Exception:
            pass

    # try to find currency or totals in workbook (fallback)
    if not currency:
        try:
            if preview_csv:
                # find currency code in preview
                m = re.search(r'\b(USD|EUR|GBP|INR|JPY|AUD|CAD)\b', preview_csv, re.IGNORECASE)
                if m:
                    currency = m.group(1).upper()
        except Exception:
            pass

    # confidence: fraction of metadata fields present + items
    meta_present = sum(1 for v in (order_id, client_name, order_date, delivery_date, order_total) if v)
    conf = 0.0
    try:
        conf = min(1.0, (meta_present / 5.0) * 0.6 + (0.4 if items else 0.0))
    except Exception:
        conf = 0.0

    # diagnostics
    meta = {
        "order_id": order_id,
        "client_name": client_name,
        "order_date": order_date,
        "delivery_date": delivery_date,
        "currency": currency,
        "order_total": order_total,
        "item_count": len(items)
    }
    save_preview = preview_csv if preview_csv else ""
    save_diag(stem, meta, preview=save_preview)

    return canonical_output(order_id=order_id, client_name=client_name,
                            order_date=order_date, delivery_date=delivery_date,
                            items=items, order_total=order_total, currency=currency,
                            confidence=round(conf,2))
