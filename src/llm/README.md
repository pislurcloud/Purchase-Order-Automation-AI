# LLM Extraction (experimental)

Scaffold for LLM-based extraction. Files:
- llm_extractor.py: orchestrator (call adapters, validate JSON)
- openai_adapter.py: OpenAI API adapter (implement real calls)
- hf_adapter.py: Hugging Face adapter (implement real calls)
- ocr_easyocr_fallback.py: easyocr wrapper (optional)

Workflow:
1. First-pass deterministic extraction (existing pipeline).
2. If low confidence, call LLM via llm_extractor.extract_with_llm(...)
3. Validate returned JSON against schema and persist.

ENV:
- For OpenAI: set OPENAI_API_KEY
- For HF: set HF_API_KEY or use local model

This is intentionally a stub; implement adapters and unit tests before enabling in production.
