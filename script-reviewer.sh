# backup current reviewer.py
#cp src/ui/reviewer.py src/ui/reviewer.py.bak

# write patched reviewer.py
cat > src/ui/reviewer.py <<'PY'
# src/ui/reviewer.py
# Reviewer UI with document preview, editable fields, editable items table, flagging and audit metadata.
# Fully replaced to include robust guess_source_file (strips _output suffixes, ignores .gitkeep, supports Excel preview).

import streamlit as st
import json, io, os, base64, time
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="PO Extraction — Reviewer UI (PoC)", layout="wide")
st.title("PO Extraction — Reviewer UI (PoC)")

# Candidate folders to find outputs and mocks
CANDIDATE_OUTPUTS = [
    Path.cwd() / "outputs",                 # repo_root/outputs
    Path.cwd() / "src" / "outputs",         # src/outputs
    Path.cwd().parent / "outputs",          # parent/outputs
    Path.cwd() / "data" / "outputs",        # data/outputs
    Path.cwd() / "data" / "mock_files",     # fallback - show mock files too (for demo)
]
CANDIDATE_MOCKS = [
    Path.cwd() / "data" / "mock_files",
    Path.cwd().parent / "data" / "mock_files",
    Path.cwd() / "mock_files",
]

def find_outputs_folder():
    for p in CANDIDATE_OUTPUTS:
        if p.exists() and any(p.glob("*.json")):
            return p
    return None

OUT = find_outputs_folder()
if not OUT:
    st.warning("No parsed JSON outputs found. I looked in these locations:")
    for p in CANDIDATE_OUTPUTS:
        st.write(f"- {p}  {'(exists)' if p.exists() else '(missing)'}")
    st.info("Run `python run_all.py` to generate outputs, then refresh.")
    st.stop()

st.write(f"Using outputs folder: `{OUT}`")

# list JSON files
json_files = sorted([p for p in OUT.glob("*.json")])
if not json_files:
    st.warning(f"No JSON files present in {OUT}")
    st.stop()

# Sidebar controls
st.sidebar.header("Reviewer controls")
reviewer_name = st.sidebar.text_input("Reviewer name", value=os.getenv("USER") or "")
auto_preview = st.sidebar.checkbox("Auto preview source file (if found)", value=True)
refresh_button = st.sidebar.button("Refresh file list")

if refresh_button:
    st.experimental_rerun()

# Select which parsed JSON to review
sel = st.selectbox("Select parsed JSON to review", json_files, format_func=lambda p: p.name)

# load JSON
try:
    parsed = json.loads(sel.read_text())
except Exception as e:
    st.error(f"Failed to read JSON: {e}")
    st.stop()

# Robust guess_source_file (full replacement)
def guess_source_file(json_path):
    """
    Robust source matcher:
      - strips common extractor suffixes from the JSON stem (e.g. _output, _output_raw, _raw, _parsed)
      - tries exact stem equality, containment, token matching, then fallback
      - ignores hidden files and .gitkeep
    """
    name = json_path.stem
    # remove common suffixes added by extractor
    for suff in ('_output_raw', '_output', '_raw', '_parsed', '_extracted'):
        if name.lower().endswith(suff):
            name = name[:-len(suff)]
            break
    name = name.strip('_').lower()

    tokens = [t for t in name.split('_') if t]
    candidate_tokens = []
    if len(tokens) >= 2:
        candidate_tokens.append(f"{tokens[0]}_{tokens[1]}")
    for t in tokens:
        if t and t.lower() != 'client' and t not in candidate_tokens:
            candidate_tokens.append(t)
    candidate_tokens.append(name)

    # scanning logic: exact -> containment -> token -> fallback
    for d in CANDIDATE_MOCKS:
        if not d.exists():
            continue
        # exact equality
        for f in d.iterdir():
            if not f.is_file():
                continue
            if f.name.startswith('.') or f.name.lower() == '.gitkeep':
                continue
            if f.stem.lower() == name:
                return f
    # containment (json base in filename or filename in base)
    for d in CANDIDATE_MOCKS:
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            if f.name.startswith('.') or f.name.lower() == '.gitkeep':
                continue
            stem = f.stem.lower()
            if name in stem or stem in name:
                return f
    # token matching
    for d in CANDIDATE_MOCKS:
        if not d.exists():
            continue
        for tok in candidate_tokens:
            for f in d.iterdir():
                if not f.is_file():
                    continue
                if f.name.startswith('.') or f.name.lower() == '.gitkeep':
                    continue
                if tok and tok.lower() in f.stem.lower():
                    return f
    # fallback: first supported type
    for d in CANDIDATE_MOCKS:
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            if f.name.startswith('.') or f.name.lower() == '.gitkeep':
                continue
            if f.suffix.lower() in ('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.xlsx', '.xls', '.csv'):
                return f
    return None

source_file = guess_source_file(sel)

# Layout: two columns, left preview, right editable fields
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Source document preview")
    if source_file and source_file.exists() and auto_preview:
        sf = source_file
        if sf.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff"):
            st.image(str(sf), caption=sf.name, use_container_width=True)
            st.markdown(f"**Source:** `{sf}`")
        elif sf.suffix.lower() in (".xlsx", ".xls", ".csv"):
            st.markdown(f"**Source (table preview):** `{sf.name}`")
            try:
                import pandas as pd
                if sf.suffix.lower() in ('.xlsx', '.xls'):
                    df = pd.read_excel(sf, sheet_name=0)
                else:
                    df = pd.read_csv(sf)
                # show first N rows for preview
                st.dataframe(df.head(20))
                st.markdown(f"Showing first 20 rows of `{sf.name}`. Download:")
                with open(sf, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{sf.name}">Download {sf.name}</a>'
                    st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.write("Failed to preview spreadsheet:", e)
                with open(sf, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{sf.name}">Download {sf.name}</a>'
                    st.markdown(href, unsafe_allow_html=True)
        elif sf.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path
                imgs = convert_from_path(str(sf), dpi=150, first_page=1, last_page=1)
                if imgs:
                    buf = io.BytesIO()
                    imgs[0].save(buf, format="PNG")
                    st.image(buf.getvalue(), caption=f"{sf.name} (page 1)", use_container_width=True)
                    st.markdown(f"**Source:** `{sf}`")
                else:
                    st.write("Could not convert PDF to image for preview.")
            except Exception as e:
                st.write("Preview unavailable (pdf2image/poppler not installed or conversion failed).")
                st.write(f"Error: {e}")
                with open(sf, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{sf.name}">Download PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
        else:
            st.write("Preview not supported for this file type.")
    else:
        st.info("No source file found for this JSON (or auto preview is disabled).")
        st.write("Searched in:")
        for d in CANDIDATE_MOCKS:
            st.write(f"- {d}  {'(exists)' if d.exists() else '(missing)'}")

    st.markdown("---")
    st.subheader("Reviewer actions")
    flagged = st.checkbox("Flag this document / require escalation", value=False)
    flag_reason = st.text_area("Flag reason / notes (if flagged)", value="")
    st.write("Reviewer notes (free text):")
    reviewer_notes = st.text_area("Notes", value="", height=120)

with col2:
    st.subheader("Parsed fields (editable)")
    # Show canonical top-level keys first (order known)
    top_keys = ['order_id', 'client_name', 'order_date', 'delivery_date', 'order_total', 'currency', 'special_instructions', 'confidence_score', 'validation_ok', 'validation_errors']
    edited = {}
    # display keys present in parsed
    for k in top_keys:
        if k in parsed:
            v = parsed.get(k)
            # for validation lists show as text for now
            if isinstance(v, list):
                edited[k] = st.text_area(k, value=json.dumps(v, indent=2), height=80)
            else:
                edited[k] = st.text_input(k, value="" if v is None else str(v))

    # Any additional keys present but not in top_keys
    others = [k for k in parsed.keys() if k not in top_keys and k != 'items']
    if others:
        st.write("Other extracted fields:")
        for k in others:
            v = parsed.get(k)
            edited[k] = st.text_input(k, value="" if v is None else str(v))

    st.markdown("---")
    st.subheader("Items (editable)")
    items = parsed.get('items', []) or []
    # convert items (list of dicts) to a list of rows suitable for data editor
    import pandas as pd
    if isinstance(items, list):
        df_items = pd.DataFrame(items)
    else:
        df_items = pd.DataFrame(items)  # fallback, may be empty
    # ensure columns exist
    for col in ['product_code','description','quantity','unit_price','total_price']:
        if col not in df_items.columns:
            df_items[col] = None
    # show data editor (Streamlit 1.18+ supports st.data_editor, older versions have experimental)
    try:
        # st.data_editor offers better UX if available
        edited_df = st.data_editor(df_items, num_rows="dynamic")
    except Exception:
        try:
            edited_df = st.experimental_data_editor(df_items)
        except Exception:
            # fallback to editing JSON text
            st.write("Your Streamlit version does not support data editor. Items shown as JSON:")
            st.write(items)
            edited_df = df_items

    st.markdown("---")
    st.write("When you click Save corrections the edited fields + items + reviewer notes will be persisted to the `labels/` folder along with a small audit record.")

    if st.button("Save corrections"):
        # Build saved object merging edited top-level fields and edited items
        saved = {}
        # coerce numeric fields back where sensible
        def coerce_val(k, v):
            if k in ('order_total','confidence_score'):
                try:
                    return float(v)
                except:
                    return v
            if k in ('quantity',):
                try:
                    return int(float(v))
                except:
                    return v
            return v

        for k, v in edited.items():
            # if text area contains JSON array (validation errors) try to parse
            if isinstance(v, str) and (v.strip().startswith('[') or v.strip().startswith('{')):
                try:
                    saved[k] = json.loads(v)
                except:
                    saved[k] = v
            else:
                saved[k] = coerce_val(k, v)

        # items: convert edited_df back to list of dicts
        try:
            items_list = edited_df.to_dict(orient='records')
        except Exception:
            # if edited_df is not a DataFrame (fallback), keep original items
            items_list = items

        saved['items'] = items_list
        saved['reviewer'] = reviewer_name or ""
        saved['flagged'] = bool(flagged)
        saved['flag_reason'] = flag_reason
        saved['reviewer_notes'] = reviewer_notes
        saved['review_timestamp'] = datetime.utcnow().isoformat() + "Z"

        # write to labels folder sibling to outputs
        labels_dir = OUT.parent / "labels"
        # if exists and is file, rename to backup
        if labels_dir.exists() and not labels_dir.is_dir():
            backup = labels_dir.with_suffix(".bak")
            labels_dir.rename(backup)
        labels_dir.mkdir(parents=True, exist_ok=True)

        outp = labels_dir / sel.name
        outp.write_text(json.dumps(saved, indent=2))
        st.success(f"Saved corrected file to {outp}")
        # append audit log entry
        audit = {
            'json': str(sel.name),
            'saved_to': str(outp),
            'timestamp': saved['review_timestamp'],
            'reviewer': saved['reviewer'],
            'flagged': saved['flagged']
        }
        (labels_dir / "audit.log").write_text(json.dumps(audit) + "\n", append=False)  # overwrite: single-entry log
        st.experimental_rerun()
PY
