# src/enhanced/ocr_easyocr_fallback.py
from pathlib import Path
_import_error_msg = None
_reader = None

def ensure_reader(lang_list=("en",), gpu=False):
    global _reader, _import_error_msg
    if _reader is not None or _import_error_msg is not None:
        return
    try:
        import easyocr
        _reader = easyocr.Reader(list(lang_list), gpu=gpu)
    except Exception as e:
        _import_error_msg = str(e)

def ocr_easyocr_as_text(path, langs=("en",), gpu=False):
    """
    Return OCR text using EasyOCR. If EasyOCR not installed, raises ImportError.
    """
    ensure_reader(langs, gpu=gpu)
    if _import_error_msg:
        raise ImportError(f"easyocr not available: {_import_error_msg}")
    result = _reader.readtext(str(path), detail=0, paragraph=True)
    # result is list of strings; join into a single block
    if isinstance(result, list):
        return "\n".join([r for r in result if r])
    return str(result)
