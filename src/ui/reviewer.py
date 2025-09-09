import streamlit as st, json
from pathlib import Path

OUT = Path('../../outputs').resolve()
LABELS = Path('../../labels').resolve()

st.title('PO Extraction — Reviewer UI (PoC)')

files = list(OUT.glob('*.json'))
sel = st.selectbox('Select parsed JSON', files)
if sel:
    data = json.loads(open(sel).read())
    st.subheader('Parsed Fields')
    edited = {}
    for k,v in data.items():
        if k == 'items':
            st.write('Items:')
            st.write(v)
            edited['items'] = v
        else:
            edited[k] = st.text_input(k, value=str(v) if v is not None else '')
    if st.button('Save corrections'):
        LABELS.mkdir(parents=True, exist_ok=True)
        outp = LABELS / sel.name
        json.dump(edited, open(outp,'w'), indent=2)
        st.success(f'Saved corrected file to {outp}')
