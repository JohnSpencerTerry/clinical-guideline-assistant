"""Urgent/emergency detection: keyword/pattern check first, LLM classifier fallback.

Biased toward over-triggering — a false positive costs a redirect, a false
negative could mean an emergency gets a calm RAG answer.
"""
