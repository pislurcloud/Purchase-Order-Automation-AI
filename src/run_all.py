import json
from pathlib import Path
OUT_DIR = Path("outputs")
MOCK_DIR = Path("data/mock_files")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_run(func, src, outname):
    try:
        res = func(src)
    except Exception as e:
        res = {'error': str(e), 'confidence_score': 0.0}
    Path(OUT_DIR / outname).write_text(json.dumps(res, indent=2))
    return res

def run():
    from src.extract_excel import extract_excel
    from src.extract_ocr import extract_from_image
    from src.extract_pdf import extract_from_pdf
    from src.transform import transform_to_schema
    from src.validate import validate_obj

    # Client B - Excel
    excel_src = MOCK_DIR / "mock_client_b_multisheet.xlsx"
    res_b = safe_run(extract_excel, str(excel_src), "client_b_output_raw.json")
    schema_b = transform_to_schema(res_b)
    ok, errors = validate_obj(schema_b)
    schema_b['validation_ok'] = ok
    schema_b['validation_errors'] = errors or schema_b.get('validation_errors',[])
    Path(OUT_DIR / "client_b_output.json").write_text(json.dumps(schema_b, indent=2))

    # Client A - PDF
    pdf_src = MOCK_DIR / "mock_client_a_table.pdf"
    res_a = safe_run(extract_from_pdf, str(pdf_src), "client_a_output_raw.json")
    schema_a = transform_to_schema(res_a)
    ok, errors = validate_obj(schema_a)
    schema_a['validation_ok'] = ok
    schema_a['validation_errors'] = errors or schema_a.get('validation_errors',[])
    Path(OUT_DIR / "client_a_output.json").write_text(json.dumps(schema_a, indent=2))

    # Client E - Scanned form
    img_src = MOCK_DIR / "mock_client_e_scanned_form.png"
    res_e = safe_run(extract_from_image, str(img_src), "client_e_output_raw.json")
    schema_e = transform_to_schema(res_e)
    ok, errors = validate_obj(schema_e)
    schema_e['validation_ok'] = ok
    schema_e['validation_errors'] = errors or schema_e.get('validation_errors',[])
    Path(OUT_DIR / "client_e_output.json").write_text(json.dumps(schema_e, indent=2))

    print("Wrote outputs to", OUT_DIR.resolve())

if __name__ == '__main__':
    run()
