"""
OpenAI adapter (scaffold).
Implement actual calls here, keep this file small and provider-specific.
Ensure you read OPENAI_API_KEY from env and never hardcode credentials.
"""

import os
def call_openai_chat(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=1500):
    """
    Placeholder: implement OpenAI API call here.
    Return a dict: {"text": "<raw model text>", "raw": <provider response>}
    """
    # Example return structure expected by orchestrator
    return {"text": None, "raw": None, "note": "openai adapter stub - implement real calls"}
