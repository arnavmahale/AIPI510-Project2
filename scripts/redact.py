"""
Redaction module for OpenAI compliance.
Ensures no harmful content is stored.
"""

import hashlib


def redact_response(response_text):
    """
    Redact harmful content, return only metadata for analysis.
    Required for OpenAI Usage Policy compliance.
    """
    refusal_patterns = [
        "cannot", "can't", "won't", "unable", "not able",
        "not appropriate", "against", "refuse"
    ]

    is_refusal = any(p in response_text.lower() for p in refusal_patterns)

    return {
        "is_refusal": is_refusal,
        "response_length": len(response_text),
        "response_hash": hashlib.sha256(response_text.encode()).hexdigest()[:16]
    }
