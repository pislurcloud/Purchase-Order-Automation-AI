# src/ui/enhanced_reviewer.py - FIXED VERSION with working save button
# Complete Enhanced Reviewer UI that uses outputs_enhanced folder and shows comparison with original
import streamlit as st
import json, io, os, base64, time
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Enhanced PO Extraction — Reviewer UI", layout="wide")
st.title("🚀 Enhanced PO Extraction — Reviewer UI")
st.caption("Showing LLM-enhanced results with comparison to original extraction")

# Updated candidate folders to prioritize enhanced outputs
CANDIDATE_OUTPUTS = [
    Path.cwd() / "outputs_enhanced",        # PRIMARY: Enhanced outputs
    Path.cwd() / "outputs",                 # FALLBACK: Original outputs
    Path.cwd() / "src" / "outputs_enhanced",
    Path.cwd() / "src" / "outputs",
    Path.cwd().parent / "outputs_enhanced",
    Path.cwd().parent / "outputs",
]

CANDIDATE_ORIGINALS = [
    Path.cwd() / "outputs",                 # Original outputs for comparison
    Path.cwd() / "src" / "outputs",
    Path.cwd().parent / "outputs",
]

CANDIDATE_MOCKS = [
    Path.cwd() / "data" / "mock_files",
    Path.cwd().parent / "data" / "mock_files",
    Path.cwd() / "mock_files",
]

def find_outputs_folder():
    """Find enhanced outputs folder first, fallback to regular outputs"""
    for p in CANDIDATE_OUTPUTS:
        if p.exists() and any(p.glob("*.json")):
            return p
    return None

def find_originals_folder():
    """Find original outputs folder for comparison"""
    for p in CANDIDATE_ORIGINALS:
        if p.exists() and any(p.glob("*.json")):
            return p
    return None

# Find both enhanced and original output folders
OUT = find_outputs_folder()
ORIG_OUT = find_originals_folder()

if not OUT:
    st.error("❌ No enhanced JSON outputs found. Please run the enhanced processing pipeline first:")
    st.code("python enhanced_run_all.py")
    st.info("Looking for outputs in these locations:")
    for p in CANDIDATE_OUTPUTS:
        st.write(f"- {p}  {'(exists)' if p.exists() else '(missing)'}")
    st.stop()

# Show which folder we're using
if "outputs_enhanced" in str(OUT):
    st.success(f"✅ Using enhanced outputs: `{OUT}`")
    if ORIG_OUT and "outputs_enhanced" not in str(ORIG_OUT):
        st.info(f"📊 Original outputs available for comparison: `{ORIG_OUT}`")
else:
    st.warning(f"⚠️ Using regular outputs (enhanced not found): `{OUT}`")

# List JSON files
json_files = sorted([p for p in OUT.glob("*.json")])
if not json_files:
    st.warning(f"No JSON files present in {OUT}")
    st.stop()

# Sidebar controls
st.sidebar.header("🔧 Reviewer Controls")
reviewer_name = st.sidebar.text_input("👤 Reviewer name", value=os.getenv("USER") or "")
auto_preview = st.sidebar.checkbox("🔍 Auto preview source file", value=True)
show_comparison = st.sidebar.checkbox("📊 Show original vs enhanced comparison", value=True)
show_llm_info = st.sidebar.checkbox("🤖 Show LLM enhancement details", value=True)
refresh_button = st.sidebar.button("🔄 Refresh file list")

if refresh_button:
    st.rerun()

# Select which parsed JSON to review
sel = st.selectbox("📁 Select enhanced extraction to review", json_files, format_func=lambda p: p.name)

# Load enhanced JSON
try:
    enhanced_data = json.loads(sel.read_text())
except Exception as e:
    st.error(f"Failed to read enhanced JSON: {e}")
    st.stop()

# Try to load original extraction for comparison
original_data = None
if ORIG_OUT and show_comparison:
    # Try to find corresponding original file
    original_candidates = [
        ORIG_OUT / sel.name,  # Same filename
        ORIG_OUT / sel.name.replace("_enhanced", "_output"),  # Replace enhanced with output
        ORIG_OUT / sel.name.replace("_enhanced", ""),  # Remove enhanced suffix
        ORIG_OUT / (sel.stem.replace("_enhanced", "") + "_output.json"),  # Construct output name
    ]
    
    for orig_path in original_candidates:
        if orig_path.exists():
            try:
                original_data = json.loads(orig_path.read_text())
                st.sidebar.success(f"✅ Original extraction loaded: {orig_path.name}")
                break
            except Exception as e:
                continue
    
    if not original_data:
        st.sidebar.warning("⚠️ No corresponding original extraction found for comparison")

def guess_source_file(json_path):
    """Robust source matcher with enhanced filename handling"""
    name = json_path.stem
    # Remove common suffixes added by extractor
    for suff in ('_enhanced', '_output_raw', '_output', '_raw', '_parsed', '_extracted'):
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

    # Exact match first
    for d in CANDIDATE_MOCKS:
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file() or f.name.startswith('.') or f.name.lower() == '.gitkeep':
                continue
            if f.stem.lower() == name:
                return f
    
    # Token matching
    for d in CANDIDATE_MOCKS:
        if not d.exists():
            continue
        for tok in candidate_tokens:
            for f in d.iterdir():
                if not f.is_file() or f.name.startswith('.') or f.name.lower() == '.gitkeep':
                    continue
                if tok and tok.lower() in f.stem.lower():
                    return f
    
    # Fallback
    for d in CANDIDATE_MOCKS:
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file() or f.name.startswith('.') or f.name.lower() == '.gitkeep':
                continue
            if f.suffix.lower() in ('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.xlsx', '.xls', '.csv', '.txt'):
                return f
    return None

source_file = guess_source_file(sel)

# Main layout
if show_comparison and original_data:
    # Three column layout: source, original, enhanced
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("📄 Source Document")
    with col2:
        st.subheader("🔧 Original Extraction")
    with col3:
        st.subheader("🚀 Enhanced Extraction")
else:
    # Two column layout: source, enhanced
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📄 Source Document")
    with col2:
        st.subheader("🚀 Enhanced Extraction")

# Source document preview (col1)
with col1:
    if source_file and source_file.exists() and auto_preview:
        sf = source_file
        if sf.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff"):
            st.image(str(sf), caption=sf.name, use_container_width=True)
        elif sf.suffix.lower() in (".xlsx", ".xls", ".csv"):
            st.caption(f"**{sf.name}**")
            try:
                import pandas as pd
                if sf.suffix.lower() in ('.xlsx', '.xls'):
                    df = pd.read_excel(sf, sheet_name=0)
                else:
                    df = pd.read_csv(sf)
                st.dataframe(df.head(10), use_container_width=True)
                
                with open(sf, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{sf.name}">⬇️ Download</a>'
                    st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Preview failed: {e}")
        elif sf.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path
                imgs = convert_from_path(str(sf), dpi=150, first_page=1, last_page=1)
                if imgs:
                    buf = io.BytesIO()
                    imgs[0].save(buf, format="PNG")
                    st.image(buf.getvalue(), caption=f"{sf.name} (page 1)", use_container_width=True)
                else:
                    st.warning("Could not convert PDF to image")
            except Exception as e:
                st.warning(f"PDF preview unavailable: {e}")
                with open(sf, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{sf.name}">Download PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
        elif sf.suffix.lower() in ('.txt',):
            st.caption(f"**{sf.name}**")
            try:
                content = sf.read_text()
                st.text_area("Content preview", content[:1000] + ("..." if len(content) > 1000 else ""), 
                           height=300, disabled=True)
            except Exception as e:
                st.error(f"Text preview failed: {e}")
        else:
            st.info(f"Preview not supported for {sf.suffix}")
    else:
        st.info("No source file found or preview disabled")
    
    # Enhancement information
    if show_llm_info and enhanced_data.get('enhancement_info'):
        st.markdown("---")
        st.subheader("🤖 LLM Enhancement Info")
        enhancement_info = enhanced_data['enhancement_info']
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Enhanced by LLM", 
                     "✅ Yes" if enhancement_info.get('enhanced_by_llm') else "❌ No")
        with col_b:
            enhancement_type = enhancement_info.get('enhancement_type', 'standard')
            st.metric("Enhancement Type", enhancement_type.replace('_', ' ').title())
        
        if enhancement_info.get('document_classification'):
            with st.expander("📋 Document Classification Details"):
                classification = enhancement_info['document_classification']
                st.json(classification)
        
        if enhancement_info.get('processing_notes'):
            st.info(f"🔍 **Processing Notes:** {enhancement_info['processing_notes']}")

# Original extraction (col2, if showing comparison)
if show_comparison and original_data:
    with col2:
        st.caption("Baseline extraction results")
        
        # Key metrics comparison
        orig_conf = original_data.get('confidence_score', 0)
        st.metric("Confidence Score", f"{orig_conf:.1%}")
        
        orig_items = len(original_data.get('items', []))
        st.metric("Items Found", orig_items)
        
        # Show key fields
        key_fields = ['order_id', 'client_name', 'order_date', 'delivery_date', 'order_total', 'currency']
        for field in key_fields:
            value = original_data.get(field)
            if value is not None:
                st.text_input(f"📝 {field}", str(value), disabled=True, key=f"orig_{field}")

# Enhanced extraction editor (col2 or col3)
edit_col = col3 if (show_comparison and original_data) else col2

with edit_col:
    st.caption("LLM-enhanced results (editable)")
    
    # Enhanced metrics
    enh_conf = enhanced_data.get('confidence_score', 0)
    orig_conf_val = original_data.get('confidence_score', 0) if original_data else 0
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.metric("Enhanced Confidence", f"{enh_conf:.1%}", 
                 delta=f"+{(enh_conf - orig_conf_val):.1%}" if original_data else None)
    with col_y:
        enh_items = len(enhanced_data.get('items', []))
        orig_items_val = len(original_data.get('items', [])) if original_data else 0
        st.metric("Items Found", enh_items, 
                 delta=f"+{enh_items - orig_items_val}" if original_data and enh_items != orig_items_val else None)
    
    # Editable fields
    edited = {}
    top_keys = ['order_id', 'client_name', 'order_date', 'delivery_date', 'order_total', 'currency', 'special_instructions']
    
    for k in top_keys:
        if k in enhanced_data:
            v = enhanced_data.get(k)
            if isinstance(v, list):
                edited[k] = st.text_area(f"✏️ {k}", value=json.dumps(v, indent=2), height=80)
            else:
                # Show field confidence if available
                field_conf = enhanced_data.get('field_confidences', {}).get(f"{k}_confidence")
                label = f"✏️ {k}"
                if field_conf:
                    label += f" ({field_conf:.1%} confidence)"
                edited[k] = st.text_input(label, value="" if v is None else str(v), key=f"enh_{k}")

    # Show other fields
    others = [k for k in enhanced_data.keys() if k not in top_keys and k != 'items' 
              and k not in ['enhancement_info', 'field_confidences', 'processing_metadata']]
    if others:
        with st.expander("📝 Other extracted fields"):
            for k in others:
                v = enhanced_data.get(k)
                edited[k] = st.text_input(f"Other: {k}", value="" if v is None else str(v), key=f"other_{k}")

# Items comparison and editing (full width)
st.markdown("---")
st.subheader("📋 Line Items Comparison & Editing")

# Initialize edited_df
edited_df = None

if show_comparison and original_data:
    tab1, tab2 = st.tabs(["🔧 Original Items", "🚀 Enhanced Items (Editable)"])
    
    with tab1:
        orig_items = original_data.get('items', [])
        if orig_items:
            import pandas as pd
            df_orig = pd.DataFrame(orig_items)
            st.dataframe(df_orig, use_container_width=True)
            st.caption(f"Found {len(orig_items)} items in original extraction")
        else:
            st.info("No items found in original extraction")
    
    with tab2:
        enh_items = enhanced_data.get('items', [])
        if enh_items:
            import pandas as pd
            df_enh = pd.DataFrame(enh_items)
            
            # Ensure required columns
            for col in ['product_code','description','quantity','unit_price','total_price']:
                if col not in df_enh.columns:
                    df_enh[col] = None
            
            try:
                edited_df = st.data_editor(df_enh, num_rows="dynamic", use_container_width=True, key="enhanced_items_editor")
                st.caption(f"✏️ Editing {len(enh_items)} enhanced items - you can add/remove rows")
            except Exception as e:
                st.warning(f"Data editor not available: {e}")
                st.dataframe(df_enh, use_container_width=True)
                edited_df = df_enh
        else:
            st.info("No items found in enhanced extraction")
            import pandas as pd
            # Create empty dataframe with required columns for editing
            df_empty = pd.DataFrame(columns=['product_code','description','quantity','unit_price','total_price'])
            try:
                edited_df = st.data_editor(df_empty, num_rows="dynamic", use_container_width=True, key="empty_items_editor")
                st.caption("✏️ No items found - you can add items manually")
            except:
                edited_df = df_empty
else:
    # Single items editor
    items = enhanced_data.get('items', [])
    if items:
        import pandas as pd
        df_items = pd.DataFrame(items)
        
        # Show item confidence scores if available
        if any('confidence' in item for item in items):
            st.caption("💡 Item confidence scores included")
        
        for col in ['product_code','description','quantity','unit_price','total_price']:
            if col not in df_items.columns:
                df_items[col] = None
        
        try:
            edited_df = st.data_editor(df_items, num_rows="dynamic", use_container_width=True, key="single_items_editor")
            st.caption(f"✏️ Editing {len(items)} items - you can add/remove rows")
        except Exception as e:
            st.warning(f"Data editor not available: {e}")
            st.dataframe(df_items, use_container_width=True)
            edited_df = df_items
    else:
        st.info("No items found")
        import pandas as pd
        # Create empty dataframe for adding items
        df_empty = pd.DataFrame(columns=['product_code','description','quantity','unit_price','total_price'])
        try:
            edited_df = st.data_editor(df_empty, num_rows="dynamic", use_container_width=True, key="add_items_editor")
            st.caption("✏️ No items found - you can add items manually")
        except:
            edited_df = df_empty

# Reviewer actions
st.markdown("---")
col_rev1, col_rev2 = st.columns(2)

with col_rev1:
    st.subheader("👤 Reviewer Actions")
    flagged = st.checkbox("🚩 Flag for escalation", value=False)
    flag_reason = st.text_area("📝 Flag reason (if flagged)", value="", height=100)

with col_rev2:
    st.subheader("📝 Review Notes")
    reviewer_notes = st.text_area("💭 Reviewer notes and feedback", value="", height=120,
                                 placeholder="Add notes about extraction quality, corrections made, or issues found...")

# SAVE BUTTON - PROMINENTLY DISPLAYED
st.markdown("---")
col_save1, col_save2, col_save3 = st.columns([1, 2, 1])

with col_save2:
    save_button = st.button("💾 **SAVE ENHANCED CORRECTIONS**", 
                           type="primary", 
                           use_container_width=True,
                           help="Save all edits and corrections to improve future LLM performance")

if save_button:
    with st.spinner("💾 Saving enhanced corrections..."):
        saved = {}
        
        # Coerce values
        def coerce_val(k, v):
            if k in ('order_total','confidence_score'):
                try:
                    return float(v) if v and str(v).strip() else None
                except:
                    return v
            if k in ('quantity',):
                try:
                    return int(float(v)) if v and str(v).strip() else None
                except:
                    return v
            return v if v and str(v).strip() else None

        # Process edited fields
        for k, v in edited.items():
            if isinstance(v, str) and (v.strip().startswith('[') or v.strip().startswith('{')):
                try:
                    saved[k] = json.loads(v)
                except:
                    saved[k] = v
            else:
                saved[k] = coerce_val(k, v)

        # Process items
        try:
            if edited_df is not None and not edited_df.empty:
                items_list = []
                for _, row in edited_df.iterrows():
                    item = {}
                    for col in ['product_code','description','quantity','unit_price','total_price']:
                        val = row.get(col)
                        if col in ['quantity']:
                            try:
                                item[col] = int(float(val)) if val and str(val).strip() else None
                            except:
                                item[col] = val
                        elif col in ['unit_price', 'total_price']:
                            try:
                                item[col] = float(val) if val and str(val).strip() else None
                            except:
                                item[col] = val
                        else:
                            item[col] = str(val) if val and str(val).strip() else None
                    # Only add items with at least some content
                    if any(item.values()):
                        items_list.append(item)
                saved['items'] = items_list
            else:
                saved['items'] = enhanced_data.get('items', [])
        except Exception as e:
            st.warning(f"Error processing items: {e}")
            saved['items'] = enhanced_data.get('items', [])

        # Add review metadata
        saved['reviewer'] = reviewer_name or ""
        saved['flagged'] = bool(flagged)
        saved['flag_reason'] = flag_reason.strip() if flag_reason else ""
        saved['reviewer_notes'] = reviewer_notes.strip() if reviewer_notes else ""
        saved['review_timestamp'] = datetime.utcnow().isoformat() + "Z"
        saved['reviewed_enhanced_version'] = True  # Mark as enhanced review
        
        # Preserve enhancement metadata
        if enhanced_data.get('enhancement_info'):
            saved['enhancement_info'] = enhanced_data['enhancement_info']
        if enhanced_data.get('field_confidences'):
            saved['field_confidences'] = enhanced_data['field_confidences']
        if enhanced_data.get('processing_metadata'):
            saved['processing_metadata'] = enhanced_data['processing_metadata']

        # Calculate improvements for audit
        original_conf = original_data.get('confidence_score', 0) if original_data else 0
        enhanced_conf = saved.get('confidence_score', enhanced_data.get('confidence_score', 0))
        
        # Save to labels folder
        labels_dir = OUT.parent / "labels_enhanced"
        labels_dir.mkdir(parents=True, exist_ok=True)

        output_path = labels_dir / sel.name
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(saved, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Failed to save file: {e}")
            st.stop()
        
        # Audit log
        audit_entry = {
            'json_file': str(sel.name),
            'saved_to': str(output_path),
            'timestamp': saved['review_timestamp'],
            'reviewer': saved['reviewer'],
            'flagged': saved['flagged'],
            'enhanced_version': True,
            'original_confidence': original_conf,
            'enhanced_confidence': enhanced_conf,
            'confidence_improvement': enhanced_conf - original_conf,
            'llm_enhanced': enhanced_data.get('enhancement_info', {}).get('enhanced_by_llm', False),
            'items_count': len(saved.get('items', [])),
            'fields_corrected': len([k for k, v in edited.items() if v and str(v).strip()])
        }
        
        audit_file = labels_dir / "enhanced_audit.log"
        try:
            with open(audit_file, "a", encoding="utf-8") as af:
                af.write(json.dumps(audit_entry) + "\n")
        except Exception as e:
            st.warning(f"Failed to write audit log: {e}")

        # Success feedback
        st.success("✅ **Enhanced corrections saved successfully!**")
        col_success1, col_success2 = st.columns(2)
        with col_success1:
            st.info(f"📁 **Saved to:** {output_path.name}")
        with col_success2:
            st.info(f"📊 **Confidence:** {enhanced_conf:.1%} (↑{enhanced_conf-original_conf:+.1%})")
        
        st.info("💡 **These corrections will be used to improve future LLM enhancements**")
        
        # Auto-refresh after short delay
        time.sleep(2)
        st.rerun()

# Show processing comparison summary
if show_comparison and original_data:
    st.markdown("---")
    st.subheader("📊 Processing Comparison Summary")
    
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    
    with col_sum1:
        improvement = enh_conf - orig_conf_val
        st.metric("Confidence Improvement", 
                 f"+{improvement:.1%}",
                 delta=f"{improvement:.1%}")
    
    with col_sum2:
        orig_fields = sum(1 for v in original_data.values() if v not in [None, '', []])
        enh_fields = sum(1 for v in enhanced_data.values() if v not in [None, '', []])
        st.metric("Fields Extracted", enh_fields, delta=f"+{enh_fields - orig_fields}")
    
    with col_sum3:
        processing_time = enhanced_data.get('processing_metadata', {}).get('processing_time_ms', 'N/A')
        st.metric("Processing Time", f"{processing_time}ms" if processing_time != 'N/A' else 'N/A')
    
    with col_sum4:
        needs_review = enhanced_data.get('processing_metadata', {}).get('needs_review', False)
        st.metric("Needs Review", "Yes" if needs_review else "No",
                 delta="Manual review required" if needs_review else "Auto-approved")

# Footer with instructions
st.markdown("---")
st.markdown("""
### 📋 **How to Use This Interface:**

1. **📖 Review** the LLM-enhanced extraction results above
2. **📊 Compare** with original extraction (if available) to see improvements  
3. **✏️ Edit** any incorrect fields or items directly in the interface
4. **📝 Add notes** about extraction quality or issues found
5. **💾 Save corrections** to improve future LLM performance

### 🎯 **Key Features:**
- **🔄 Live comparison** between original and enhanced results
- **📈 Confidence scoring** with improvement tracking  
- **🤖 LLM enhancement details** showing processing decisions
- **🎯 Field-level confidence** indicators for quality assessment
- **🧠 Correction learning** system for continuous improvement

### 💡 **Tips:**
- Use the **data editor** to add/remove/edit line items
- **Field confidence scores** help identify uncertain extractions
- **Flag documents** that need escalation or have complex issues
- **Detailed notes** help improve future processing accuracy
""")

# Debug information (collapsible)
with st.expander("🔧 Debug Information"):
    st.write("**File paths:**")
    st.code(f"Enhanced output: {OUT}")
    st.code(f"Original output: {ORIG_OUT}")
    st.code(f"Source file: {source_file}")
    
    st.write("**Enhancement info:**")
    if enhanced_data.get('enhancement_info'):
        st.json(enhanced_data['enhancement_info'])
    else:
        st.write("No enhancement info available")
        
    st.write("**Processing metadata:**")
    if enhanced_data.get('processing_metadata'):
        st.json(enhanced_data['processing_metadata'])
    else:
        st.write("No processing metadata available")