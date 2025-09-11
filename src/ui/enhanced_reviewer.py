# src/ui/enhanced_reviewer.py - FIXED VERSION with proper original vs enhanced comparison
# Complete Enhanced Reviewer UI that properly compares enhanced results with original extraction
import streamlit as st
import json, io, os, base64, time
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Enhanced PO Extraction — Reviewer UI", layout="wide")
st.title("🚀 Enhanced PO Extraction — Reviewer UI")
st.caption("Comparing LLM-enhanced results with original extraction")

# Define candidate folders with better organization
CANDIDATE_ENHANCED = [
    Path.cwd() / "outputs_enhanced_robust",   # PRIMARY: New robust enhanced outputs
    Path.cwd() / "outputs_enhanced",          # SECONDARY: Regular enhanced outputs
    Path.cwd() / "src" / "outputs_enhanced_robust",
    Path.cwd() / "src" / "outputs_enhanced",
    Path.cwd().parent / "outputs_enhanced_robust",
    Path.cwd().parent / "outputs_enhanced",
]

CANDIDATE_ORIGINALS = [
    Path.cwd() / "outputs",                   # PRIMARY: Original baseline outputs
    Path.cwd() / "src" / "outputs",
    Path.cwd().parent / "outputs",
    Path.cwd() / "data" / "outputs",
]

CANDIDATE_MOCKS = [
    Path.cwd() / "data" / "mock_files",
    Path.cwd().parent / "data" / "mock_files",
    Path.cwd() / "mock_files",
]

def find_outputs_folder(candidates, description):
    """Find the first existing folder with JSON files"""
    for p in candidates:
        if p.exists() and any(p.glob("*.json")):
            return p
    return None

def find_corresponding_original(enhanced_file_path, original_dirs):
    """
    Find the corresponding original file for an enhanced extraction.
    Handles various naming conventions:
    - client_a_enhanced_robust.json -> client_a_baseline.json / client_a_output.json
    - some_file_enhanced.json -> some_file_baseline.json / some_file_output.json
    """
    enhanced_name = enhanced_file_path.stem
    
    # Remove enhanced suffixes to get base name
    base_name = enhanced_name
    for suffix in ['_enhanced_robust', '_enhanced', '_robust']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    # Try multiple original naming patterns
    original_patterns = [
        f"{base_name}_baseline.json",      # Standard baseline naming
        f"{base_name}_output.json",        # Standard output naming  
        f"{base_name}.json",               # Direct name match
        f"{base_name}_original.json",      # Explicit original naming
    ]
    
    # Search in all original directories
    for orig_dir in original_dirs:
        if not orig_dir.exists():
            continue
            
        for pattern in original_patterns:
            candidate = orig_dir / pattern
            if candidate.exists():
                return candidate
    
    # Fallback: look for any file with the base name
    for orig_dir in original_dirs:
        if not orig_dir.exists():
            continue
            
        for file in orig_dir.glob("*.json"):
            file_base = file.stem
            # Remove common suffixes from original files too
            for suffix in ['_output', '_baseline', '_raw', '_parsed']:
                if file_base.endswith(suffix):
                    file_base = file_base[:-len(suffix)]
                    break
            
            if file_base == base_name:
                return file
    
    return None

# Find enhanced and original output folders
ENHANCED_OUT = find_outputs_folder(CANDIDATE_ENHANCED, "enhanced")
ORIGINAL_OUT = find_outputs_folder(CANDIDATE_ORIGINALS, "original")

# Check if we have enhanced outputs
if not ENHANCED_OUT:
    st.error("❌ No enhanced JSON outputs found. Please run the enhanced processing pipeline first:")
    st.code("python enhanced_run_all_robust.py")
    st.info("Looking for enhanced outputs in these locations:")
    for p in CANDIDATE_ENHANCED:
        st.write(f"- {p}  {'✅ exists' if p.exists() else '❌ missing'}")
    st.stop()

# Show which folders we're using
st.success(f"✅ Enhanced outputs found: `{ENHANCED_OUT}`")

if ORIGINAL_OUT:
    st.info(f"📊 Original outputs found: `{ORIGINAL_OUT}`")
    comparison_available = True
else:
    st.warning("⚠️ Original outputs not found - comparison will be limited")
    st.info("Run `python run_all.py` to generate baseline outputs for comparison")
    comparison_available = False

# List enhanced JSON files
enhanced_files = sorted([p for p in ENHANCED_OUT.glob("*.json")])
if not enhanced_files:
    st.warning(f"No JSON files present in {ENHANCED_OUT}")
    st.stop()

# Sidebar controls
st.sidebar.header("🔧 Reviewer Controls")
reviewer_name = st.sidebar.text_input("👤 Reviewer name", value=os.getenv("USER") or "Anonymous")
auto_preview = st.sidebar.checkbox("🔍 Auto preview source file", value=True)
show_comparison = st.sidebar.checkbox("📊 Show original vs enhanced comparison", value=comparison_available)
show_llm_info = st.sidebar.checkbox("🤖 Show LLM enhancement details", value=True)
show_debug = st.sidebar.checkbox("🔧 Show debug information", value=False)
refresh_button = st.sidebar.button("🔄 Refresh file list")

if refresh_button:
    st.rerun()

# File selection
st.subheader("📁 File Selection")
selected_file = st.selectbox(
    "Select enhanced extraction to review", 
    enhanced_files, 
    format_func=lambda p: f"{p.name} ({p.stat().st_size} bytes)"
)

# Load enhanced JSON
try:
    enhanced_data = json.loads(selected_file.read_text(encoding='utf-8'))
    st.sidebar.success(f"✅ Enhanced file loaded: {selected_file.name}")
except Exception as e:
    st.error(f"❌ Failed to read enhanced JSON: {e}")
    st.stop()

# Try to load corresponding original extraction
original_data = None
original_file = None

if comparison_available and show_comparison:
    original_dirs = [ORIGINAL_OUT] if ORIGINAL_OUT else []
    original_file = find_corresponding_original(selected_file, original_dirs)
    
    if original_file and original_file.exists():
        try:
            original_data = json.loads(original_file.read_text(encoding='utf-8'))
            st.sidebar.success(f"✅ Original file found: {original_file.name}")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to read original file: {e}")
            original_data = None
    else:
        st.sidebar.warning("⚠️ No corresponding original file found")
        if show_debug:
            st.sidebar.write("Searched for:")
            base_name = selected_file.stem.replace('_enhanced_robust', '').replace('_enhanced', '')
            for pattern in [f"{base_name}_baseline.json", f"{base_name}_output.json", f"{base_name}.json"]:
                st.sidebar.write(f"- {pattern}")

def guess_source_file(json_path):
    """Enhanced source file matching"""
    name = json_path.stem
    # Remove enhanced suffixes
    for suffix in ['_enhanced_robust', '_enhanced', '_output', '_baseline', '_raw', '_parsed']:
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    name = name.strip('_').lower()
    tokens = [t for t in name.split('_') if t and t != 'client']
    
    # Try exact match first
    for mock_dir in CANDIDATE_MOCKS:
        if not mock_dir.exists():
            continue
        for file in mock_dir.iterdir():
            if (file.is_file() and 
                not file.name.startswith('.') and 
                file.name.lower() != '.gitkeep' and
                file.stem.lower() == name):
                return file
    
    # Try token matching
    for mock_dir in CANDIDATE_MOCKS:
        if not mock_dir.exists():
            continue
        for token in tokens:
            for file in mock_dir.iterdir():
                if (file.is_file() and 
                    not file.name.startswith('.') and 
                    file.name.lower() != '.gitkeep' and
                    token in file.stem.lower() and
                    file.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.xlsx', '.xls', '.csv', '.txt']):
                    return file
    
    return None

source_file = guess_source_file(selected_file)

# Main layout based on comparison availability
if show_comparison and original_data:
    st.subheader("📊 Three-Way Comparison View")
    col1, col2, col3 = st.columns([1, 1, 1])
    headers = ["📄 Source Document", "🔧 Original Extraction", "🚀 Enhanced Extraction"]
else:
    st.subheader("📋 Enhanced Extraction View")
    col1, col2 = st.columns([1.2, 1])
    headers = ["📄 Source Document", "🚀 Enhanced Extraction"]

# Column 1: Source Document Preview
with col1:
    st.markdown(f"### {headers[0]}")
    
    if source_file and source_file.exists() and auto_preview:
        st.success(f"📎 **Source:** `{source_file.name}`")
        
        try:
            if source_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff"]:
                st.image(str(source_file), caption=source_file.name, use_container_width=True)
                
            elif source_file.suffix.lower() in [".xlsx", ".xls", ".csv"]:
                try:
                    import pandas as pd
                    if source_file.suffix.lower() in ['.xlsx', '.xls']:
                        df = pd.read_excel(source_file, sheet_name=0)
                    else:
                        df = pd.read_csv(source_file)
                    
                    st.dataframe(df.head(10), use_container_width=True)
                    st.caption(f"First 10 rows of {source_file.name}")
                    
                    # Download link
                    with open(source_file, "rb") as f:
                        data = f.read()
                        b64 = base64.b64encode(data).decode()
                        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{source_file.name}">⬇️ Download Original</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Failed to preview: {e}")
                    
            elif source_file.suffix.lower() == ".pdf":
                try:
                    from pdf2image import convert_from_path
                    imgs = convert_from_path(str(source_file), dpi=150, first_page=1, last_page=1)
                    if imgs:
                        buf = io.BytesIO()
                        imgs[0].save(buf, format="PNG")
                        st.image(buf.getvalue(), caption=f"{source_file.name} (page 1)", use_container_width=True)
                    else:
                        st.warning("Could not convert PDF to image")
                except ImportError:
                    st.warning("PDF preview requires pdf2image package")
                except Exception as e:
                    st.warning(f"PDF preview failed: {e}")
                    
            elif source_file.suffix.lower() in ['.txt']:
                try:
                    content = source_file.read_text(encoding='utf-8')
                    st.text_area("Content preview", content[:1000] + ("..." if len(content) > 1000 else ""), 
                               height=300, disabled=True)
                except Exception as e:
                    st.error(f"Text preview failed: {e}")
            else:
                st.info(f"Preview not supported for {source_file.suffix}")
                
        except Exception as e:
            st.error(f"Preview error: {e}")
    else:
        st.info("🔍 No source file found or preview disabled")
        if show_debug:
            st.write("**Debug: Source file search:**")
            for mock_dir in CANDIDATE_MOCKS:
                st.write(f"- {mock_dir} {'✅' if mock_dir.exists() else '❌'}")

# Column 2: Original Extraction (if showing comparison)
if show_comparison and original_data:
    with col2:
        st.markdown(f"### {headers[1]}")
        st.caption("Baseline extraction results")
        
        # Key metrics
        orig_conf = original_data.get('confidence_score', 0)
        orig_items = len(original_data.get('items', []))
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Confidence", f"{orig_conf:.1%}")
        with col2b:
            st.metric("Items", orig_items)
        
        # Key fields display
        key_fields = ['order_id', 'client_name', 'order_date', 'delivery_date', 'order_total', 'currency']
        
        for field in key_fields:
            value = original_data.get(field)
            if value is not None:
                st.text_input(
                    f"📝 {field.replace('_', ' ').title()}", 
                    str(value), 
                    disabled=True, 
                    key=f"orig_{field}",
                    help="Original extraction result"
                )
        
        # Show items summary
        if original_data.get('items'):
            with st.expander(f"📋 Items ({len(original_data['items'])})"):
                import pandas as pd
                try:
                    df_orig = pd.DataFrame(original_data['items'])
                    st.dataframe(df_orig, use_container_width=True)
                except Exception:
                    st.json(original_data['items'])

# Column 3: Enhanced Extraction (or Column 2 if no comparison)
edit_col = col3 if (show_comparison and original_data) else col2

with edit_col:
    st.markdown(f"### {headers[-1]}")
    st.caption("LLM-enhanced results (editable)")
    
    # Enhanced metrics with comparison
    enh_conf = enhanced_data.get('confidence_score', 0)
    enh_items = len(enhanced_data.get('items', []))
    
    if original_data:
        orig_conf_val = original_data.get('confidence_score', 0)
        orig_items_val = len(original_data.get('items', []))
        
        col3a, col3b = st.columns(2)
        with col3a:
            improvement = enh_conf - orig_conf_val
            st.metric("Enhanced Confidence", f"{enh_conf:.1%}", 
                     delta=f"{improvement:+.1%}")
        with col3b:
            item_improvement = enh_items - orig_items_val
            st.metric("Enhanced Items", enh_items, 
                     delta=f"{item_improvement:+d}" if item_improvement != 0 else None)
    else:
        col3a, col3b = st.columns(2)
        with col3a:
            st.metric("Confidence", f"{enh_conf:.1%}")
        with col3b:
            st.metric("Items", enh_items)
    
    # Editable fields
    edited = {}
    top_keys = ['order_id', 'client_name', 'order_date', 'delivery_date', 'order_total', 'currency', 'special_instructions']
    
    for k in top_keys:
        if k in enhanced_data:
            v = enhanced_data.get(k)
            
            # Show field confidence if available
            field_conf = enhanced_data.get('field_confidences', {}).get(f"{k}_confidence")
            label = f"✏️ {k.replace('_', ' ').title()}"
            if field_conf:
                label += f" ({field_conf:.1%} confidence)"
            
            # Show comparison with original if available
            help_text = "Enhanced extraction result"
            if original_data and k in original_data:
                orig_val = original_data.get(k)
                if str(orig_val) != str(v):
                    help_text += f" | Original: {orig_val}"
            
            if isinstance(v, list):
                edited[k] = st.text_area(label, value=json.dumps(v, indent=2), height=80, help=help_text)
            else:
                edited[k] = st.text_input(label, value="" if v is None else str(v), key=f"enh_{k}", help=help_text)

# LLM Enhancement Information
if show_llm_info and enhanced_data.get('enhancement_info'):
    st.markdown("---")
    st.subheader("🤖 LLM Enhancement Details")
    
    enhancement_info = enhanced_data['enhancement_info']
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        enhanced_by_llm = enhancement_info.get('enhanced_by_llm', False)
        st.metric("LLM Enhanced", "✅ Yes" if enhanced_by_llm else "❌ No")
    
    with col_info2:
        enhancement_type = enhancement_info.get('enhancement_type', 'standard')
        st.metric("Enhancement Type", enhancement_type.replace('_', ' ').title())
    
    with col_info3:
        success_rate = enhancement_info.get('success_rate', 0)
        if success_rate:
            st.metric("Success Rate", f"{success_rate:.1%}")
    
    # Strategy details
    strategies = enhancement_info.get('enhancement_strategies', [])
    if strategies:
        with st.expander("🔧 Enhancement Strategies Applied"):
            for i, strategy in enumerate(strategies, 1):
                status = "✅" if "success" in strategy else "⚠️" if "failed" in strategy else "ℹ️"
                st.write(f"{i}. {status} {strategy}")
    
    # Processing notes
    if enhancement_info.get('processing_notes'):
        st.info(f"🔍 **Processing Notes:** {enhancement_info['processing_notes']}")

# Items comparison and editing
st.markdown("---")
st.subheader("📋 Line Items Comparison & Editing")

# Initialize edited_df
edited_df = None

if show_comparison and original_data:
    # Two tabs for comparison
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
                edited_df = st.data_editor(df_enh, num_rows="dynamic", use_container_width=True, 
                                         key="enhanced_items_editor")
                st.caption(f"✏️ Editing {len(enh_items)} enhanced items")
                
                # Show changes from original
                if orig_items:
                    improvements = len(enh_items) - len(orig_items)
                    if improvements > 0:
                        st.success(f"🎯 {improvements} additional items found by LLM enhancement")
                    elif improvements < 0:
                        st.warning(f"⚠️ {abs(improvements)} fewer items than original")
                    else:
                        st.info("📊 Same number of items as original")
                        
            except Exception as e:
                st.warning(f"Data editor not available: {e}")
                st.dataframe(df_enh, use_container_width=True)
                edited_df = df_enh
        else:
            st.info("No items found in enhanced extraction")
            import pandas as pd
            df_empty = pd.DataFrame(columns=['product_code','description','quantity','unit_price','total_price'])
            try:
                edited_df = st.data_editor(df_empty, num_rows="dynamic", use_container_width=True, 
                                         key="empty_items_editor")
                st.caption("✏️ No items found - you can add items manually")
            except:
                edited_df = df_empty
else:
    # Single items editor without comparison
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
            edited_df = st.data_editor(df_items, num_rows="dynamic", use_container_width=True, 
                                     key="single_items_editor")
            st.caption(f"✏️ Editing {len(items)} items")
        except Exception as e:
            st.warning(f"Data editor not available: {e}")
            st.dataframe(df_items, use_container_width=True)
            edited_df = df_items
    else:
        st.info("No items found")
        import pandas as pd
        df_empty = pd.DataFrame(columns=['product_code','description','quantity','unit_price','total_price'])
        try:
            edited_df = st.data_editor(df_empty, num_rows="dynamic", use_container_width=True, 
                                     key="add_items_editor")
            st.caption("✏️ No items found - you can add items manually")
        except:
            edited_df = df_empty

# Reviewer actions and save functionality
st.markdown("---")
col_rev1, col_rev2 = st.columns(2)

with col_rev1:
    st.subheader("👤 Reviewer Actions")
    flagged = st.checkbox("🚩 Flag for escalation", value=False)
    flag_reason = st.text_area("📝 Flag reason (if flagged)", value="", height=100,
                              placeholder="Describe why this document needs escalation...")

with col_rev2:
    st.subheader("📝 Review Notes")
    reviewer_notes = st.text_area("💭 Reviewer notes and feedback", value="", height=120,
                                 placeholder="Add notes about extraction quality, corrections made, or issues found...")

# SAVE BUTTON
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
        
        # Process edited fields
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
                    
                    if any(item.values()):
                        items_list.append(item)
                saved['items'] = items_list
            else:
                saved['items'] = enhanced_data.get('items', [])
        except Exception as e:
            st.warning(f"Error processing items: {e}")
            saved['items'] = enhanced_data.get('items', [])

        # Add review metadata
        saved['reviewer'] = reviewer_name or "Anonymous"
        saved['flagged'] = bool(flagged)
        saved['flag_reason'] = flag_reason.strip() if flag_reason else ""
        saved['reviewer_notes'] = reviewer_notes.strip() if reviewer_notes else ""
        saved['review_timestamp'] = datetime.utcnow().isoformat() + "Z"
        saved['reviewed_enhanced_version'] = True
        
        # Preserve enhancement metadata
        for key in ['enhancement_info', 'field_confidences', 'processing_metadata']:
            if enhanced_data.get(key):
                saved[key] = enhanced_data[key]

        # Calculate improvement metrics
        original_conf = original_data.get('confidence_score', 0) if original_data else 0
        enhanced_conf = saved.get('confidence_score', enhanced_data.get('confidence_score', 0))
        
        # Save to labels folder
        labels_dir = ENHANCED_OUT.parent / "labels_enhanced_robust"
        labels_dir.mkdir(parents=True, exist_ok=True)

        output_path = labels_dir / selected_file.name
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(saved, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Failed to save file: {e}")
            st.stop()
        
        # Audit log with comparison metrics
        audit_entry = {
            'enhanced_file': str(selected_file.name),
            'original_file': str(original_file.name) if original_file else None,
            'saved_to': str(output_path),
            'timestamp': saved['review_timestamp'],
            'reviewer': saved['reviewer'],
            'flagged': saved['flagged'],
            'comparison_available': original_data is not None,
            'original_confidence': original_conf,
            'enhanced_confidence': enhanced_conf,
            'confidence_improvement': enhanced_conf - original_conf,
            'original_items': len(original_data.get('items', [])) if original_data else 0,
            'enhanced_items': len(saved.get('items', [])),
            'llm_enhanced': enhanced_data.get('enhancement_info', {}).get('enhanced_by_llm', False),
            'fields_corrected': len([k for k, v in edited.items() if v and str(v).strip()]),
            'enhancement_strategies': enhanced_data.get('enhancement_info', {}).get('enhancement_strategies', [])
        }
        
        audit_file = labels_dir / "enhanced_audit_robust.log"
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
            if original_data:
                st.info(f"📊 **Improvement:** {enhanced_conf-original_conf:+.1%} confidence")
        with col_success2:
            st.info(f"🎯 **Final Confidence:** {enhanced_conf:.1%}")
            if original_data:
                item_improvement = len(saved.get('items', [])) - len(original_data.get('items', []))
                if item_improvement > 0:
                    st.info(f"📋 **Items:** +{item_improvement} additional")
        
        st.info("💡 **These corrections will improve future LLM enhancements**")
        
        time.sleep(2)
        st.rerun()

# Processing comparison summary
if show_comparison and original_data:
    st.markdown("---")
    st.subheader("📊 Processing Comparison Summary")
    
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    
    with col_sum1:
        improvement = enh_conf - original_data.get('confidence_score', 0)
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

# Footer with enhanced instructions
st.markdown("---")
st.markdown("""
### 📋 **Enhanced Reviewer Interface Guide:**

#### 🎯 **Key Features:**
- **🔄 Live comparison** between original baseline and LLM-enhanced results
- **📈 Confidence tracking** with improvement metrics
- **🤖 LLM enhancement details** showing applied strategies and success rates
- **🎯 Field-level confidence** indicators for quality assessment
- **🧠 Learning system** - corrections improve future processing

#### 📖 **How to Use:**
1. **📊 Compare** original vs enhanced results to see LLM improvements
2. **✏️ Edit** any incorrect fields or items directly in the interface
3. **📝 Add notes** about extraction quality, issues, or suggestions
4. **💾 Save corrections** - these feed back into the learning system

#### 🎚️ **Quality Indicators:**
- **Confidence scores** show extraction certainty (aim for >80%)
- **Green metrics** indicate improvements from LLM enhancement
- **Field confidence** helps identify uncertain extractions
- **Enhancement strategies** show what processing was applied

#### 💡 **Pro Tips:**
- **Flag documents** with complex layouts or unusual formats
- **Detailed notes** help improve future processing accuracy
- **Item editing** supports adding/removing/modifying line items
- **Comparison view** helps validate LLM improvements

#### 🔧 **Troubleshooting:**
- If original comparison missing, run `python run_all.py` first
- If source preview fails, check file permissions and dependencies
- Data editor requires modern Streamlit version (1.18+)
""")

# Debug section
if show_debug:
    with st.expander("🔧 Debug Information"):
        st.write("**File Paths:**")
        st.code(f"Enhanced output dir: {ENHANCED_OUT}")
        st.code(f"Original output dir: {ORIGINAL_OUT}")
        st.code(f"Selected enhanced: {selected_file}")
        st.code(f"Found original: {original_file}")
        st.code(f"Source file: {source_file}")
        
        st.write("**Search Results:**")
        if ENHANCED_OUT:
            enhanced_count = len(list(ENHANCED_OUT.glob("*.json")))
            st.write(f"Enhanced files found: {enhanced_count}")
        
        if ORIGINAL_OUT:
            original_count = len(list(ORIGINAL_OUT.glob("*.json")))
            st.write(f"Original files found: {original_count}")
        
        st.write("**File Matching Logic:**")
        base_name = selected_file.stem.replace('_enhanced_robust', '').replace('_enhanced', '')
        st.write(f"Base name extracted: `{base_name}`")
        
        st.write("**Enhancement Info:**")
        if enhanced_data.get('enhancement_info'):
            st.json(enhanced_data['enhancement_info'])
        else:
            st.write("No enhancement info available")
            
        st.write("**Processing Metadata:**")
        if enhanced_data.get('processing_metadata'):
            st.json(enhanced_data['processing_metadata'])
        else:
            st.write("No processing metadata available")

# Performance metrics footer
if enhanced_data.get('enhancement_info'):
    enhancement_info = enhanced_data['enhancement_info']
    if enhancement_info.get('enhanced_by_llm'):
        st.markdown("---")
        st.info("🤖 **This document was processed using LLM enhancement** - improvements shown above")
        
        if original_data:
            orig_time = "N/A"  # Original processing time not tracked
            enh_time = enhanced_data.get('processing_metadata', {}).get('processing_time_ms', 'N/A')
            
            if enh_time != 'N/A':
                st.caption(f"⚡ Enhanced processing completed in {enh_time}ms")
            
            strategies = enhancement_info.get('enhancement_strategies', [])
            success_count = len([s for s in strategies if 'success' in s.lower()])
            total_count = len(strategies)
            
            if total_count > 0:
                st.caption(f"🎯 Enhancement success rate: {success_count}/{total_count} strategies successful")
    else:
        st.caption("ℹ️ This document was processed without LLM enhancement")