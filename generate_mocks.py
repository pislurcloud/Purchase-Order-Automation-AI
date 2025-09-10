#!/usr/bin/env python3
# generate_mocks.py
# Create synthetic sample files for Type A (PDF table), Type B (Excel multi-sheet), Type C (scanned form image)
# Writes output to data/mock_files/

import random
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# PDF creation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Image creation
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path("data/mock_files")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# helper data
CLIENTS = [
    ("InnovaTech Systems", "USD"),
    ("NextGen Solutions", "EUR"),
    ("Acme Corp", "USD"),
    ("Beta Ltd", "GBP"),
    ("Visionary Designs", "USD"),
    ("TechCorp Solutions", "USD"),
]

PRODUCT_POOL = [
    ("PR-001","Widget Pro"), ("PR-002","Gadget Max"), ("PR-010","Alpha Widget"),
    ("NG-010","Turbo Gear"), ("IT-001","Alpha Widget"), ("X100","X100 Device"),
    ("Y500","Y500 Module"), ("V100","Optic Lens"), ("V200","Frame Kit")
]

def rand_date(base_days_offset=0):
    base = datetime.utcnow().date() - timedelta(days=base_days_offset)
    d = base + timedelta(days=random.randint(0, 60))
    return d.strftime("%Y-%m-%d")

# --- Type A: PDF table style ---
def make_pdf_table(path: Path, order_id, client, items, order_date, delivery_date, currency):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    x = 50
    y = height - 60

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, f"{client} - Purchase Order")
    y -= 24
    c.setFont("Helvetica", 11)
    c.drawString(x, y, f"Order ID: {order_id}")
    y -= 16
    c.drawString(x, y, f"Order Date: {order_date}   Delivery Date: {delivery_date}")
    y -= 16
    c.drawString(x, y, f"Currency: {currency}")
    y -= 24

    # header
    c.setFont("Helvetica-Bold", 10)
    header = f"{'Product Code':<14}{'Description':<30}{'Quantity':>10}{'Unit Price':>15}{'Total Price':>15}"
    c.drawString(x, y, header)
    y -= 14
    c.line(x, y+6, width - 50, y+6)
    c.setFont("Helvetica", 10)

    total = 0.0
    for code, desc, qty, unit in items:
        tot = qty * unit
        total += tot
        # format columns with spacing so simple text-table parsers can pick them
        line = f"{code:<14}{desc:<30}{qty:>10}{unit:>15.2f}{tot:>15.2f}"
        if y < 80:
            c.showPage()
            y = height - 60
        c.drawString(x, y, line)
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, f"Order Total: {total:.2f}")
    c.save()

# --- Type B: Excel multi-sheet ---
def make_excel_multisheet(path: Path, orders, items):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        orders_df = pd.DataFrame(orders)
        items_df = pd.DataFrame(items)
        orders_df.to_excel(writer, sheet_name="Orders", index=False)
        items_df.to_excel(writer, sheet_name="Items", index=False)

# --- Type C: scanned form image (PNG) ---
def make_scanned_form(path: Path, order_id, client, items, total, date, currency):
    W, H = 900, 700
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    y = 30
    d.text((30, y), f"{client} - Purchase Order Form", font=font_bold, fill="black")
    y += 36
    d.text((30, y), f"Order ID: {order_id}", font=font, fill="black"); y += 28
    d.text((30, y), f"Date: {date}", font=font, fill="black"); y += 24
    d.text((30, y), f"Currency: {currency}", font=font, fill="black"); y += 26
    d.text((30, y), "Items:", font=font_bold, fill="black"); y += 28

    for code, desc, qty, price in items:
        d.text((50, y), f"{code}   {desc}   {qty} x {price:.2f}", font=font, fill="black")
        y += 22
        if y > H - 120:
            break

    y += 10
    d.text((30, y), f"Total: {total:.2f}", font=font_bold, fill="black")
    img.save(path)

# --- Utilities to generate sample sets ---
def sample_items_for_pdf(n=2):
    chosen = random.sample(PRODUCT_POOL, n)
    out = []
    for code, desc in chosen:
        qty = random.randint(1, 50)
        unit = round(random.uniform(10.0, 200.0), 2)
        out.append((code, desc, qty, unit))
    return out

def generate_samples(n_pdf=2, n_excel=2, n_img=2):
    created = []
    # PDFs - Type A
    for i in range(n_pdf):
        client, currency = random.choice(CLIENTS)
        order_id = f"PO-{random.randint(2025, 2029)}-{random.randint(1000,9999)}"
        order_date = rand_date(30)
        delivery_date = (datetime.strptime(order_date, "%Y-%m-%d") + timedelta(days=random.randint(3, 21))).strftime("%Y-%m-%d")
        items = sample_items_for_pdf(random.randint(1,3))
        fname = OUT_DIR / f"mock_client_a_sample_auto_{i+1}.pdf"
        make_pdf_table(fname, order_id, client, items, order_date, delivery_date, currency)
        created.append(fname)

    # Excels - Type B
    for i in range(n_excel):
        # create 1-2 Orders and corresponding Items
        orders = []
        items = []
        for j in range(1, random.randint(2,4)):
            order_id = f"B-{random.randint(2000,9999)}"
            date = rand_date(60)
            customer = random.choice([c for c,_ in CLIENTS])[0] if isinstance(random.choice(CLIENTS), tuple) else random.choice(CLIENTS)
            currency = random.choice(["USD","EUR","GBP","INR"])
            orders.append({"OrderID": order_id, "Date": date, "Customer": customer, "Currency": currency})
            # items for this order
            for k in range(random.randint(1,3)):
                prod = random.choice(PRODUCT_POOL)
                qty = random.randint(1,30)
                unit = round(random.uniform(5.0, 300.0), 2)
                items.append({"OrderID": order_id, "Product": prod[0], "Description": prod[1], "Qty": qty, "UnitPrice": unit, "Total": round(qty*unit,2)})
        fname = OUT_DIR / f"mock_client_b_sample_auto_{i+1}.xlsx"
        make_excel_multisheet(fname, orders, items)
        created.append(fname)

    # Images - Type C (scanned)
    for i in range(n_img):
        client, currency = random.choice(CLIENTS)
        order_id = f"C-{random.randint(2025,9999)}"
        date = rand_date(45)
        items = sample_items_for_pdf(random.randint(1,4))
        total = sum(qty*unit for (_,_,qty,unit) in items)
        fname = OUT_DIR / f"mock_client_c_sample_auto_{i+1}.png"
        make_scanned_form(fname, order_id, client, items, total, date, currency)
        created.append(fname)

    return created

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic mock files for PoC")
    parser.add_argument("--pdf", type=int, default=2, help="Number of PDF samples to generate (Type A)")
    parser.add_argument("--excel", type=int, default=2, help="Number of Excel samples to generate (Type B)")
    parser.add_argument("--img", type=int, default=2, help="Number of scanned-form image samples to generate (Type C)")
    args = parser.parse_args()
    files = generate_samples(n_pdf=args.pdf, n_excel=args.excel, n_img=args.img)
    print("Created files:")
    for f in files:
        print(" -", f)
