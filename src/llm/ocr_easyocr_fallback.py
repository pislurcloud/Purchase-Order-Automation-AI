"""
Optional wrapper for EasyOCR. The main extractor will try to import this if EasyOCR is installed.
This wrapper is intentionally safe: if easyocr is not available, it raises ImportError.
"""

def ocr_easyocr_as_text(path, langs=("en",), gpu=False):
    try:
        import easyocr
    except Exception as e:
        raise ImportError("easyocr not installed: " + str(e))
    reader = easyocr.Reader(list(langs), gpu=gpu)
    result = reader.readtext(str(path), detail=0, paragraph=True)
    return "\\n".join([r for r in result if r])
