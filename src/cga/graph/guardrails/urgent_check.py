"""Urgent/emergency detection: keyword/pattern check first, LLM classifier fallback.

Biased toward over-triggering — a false positive costs a redirect, a false
negative could mean an emergency gets a calm RAG answer.
"""

import re

from langchain_core.language_models.chat_models import BaseChatModel

_URGENT_PATTERNS = [
    r"\bchest pain\b",
    r"\b(can'?t|cannot|couldn'?t|haven'?t been able to) (breathe|catch (my|his|her|their) breath)\b",
    r"\bloss of consciousness\b",
    r"\b(passed out|passing out|fainted|unconscious)\b",
    r"\b(suicid|self.?harm|kill (myself|himself|herself|themselves))\b",
    r"\bdka\b",
    r"\bdiabetic ketoacidosis\b",
    r"\b(blood sugar|glucose) (of |is |was )?(over|above)?\s?(400|500|600)\b",
    r"\bblood sugar (below|under) (40|30|20)\b",
    r"\bsevere(ly)? (confus|disorient)",
    r"\bcalling 911\b|\bcall(ing)? an ambulance\b",
]

_URGENT_RE = re.compile("|".join(_URGENT_PATTERNS), re.IGNORECASE)

_LLM_CLASSIFIER_PROMPT = """Does the following message describe a potential medical emergency requiring \
immediate care (e.g. severe/atypical symptoms, loss of consciousness, chest pain, signs of DKA, severe \
hypo/hyperglycemia, self-harm)? Answer with exactly one word: "yes" or "no". Bias toward "yes" if unsure.

Message: {message}
"""


def keyword_check(message: str) -> bool:
    return bool(_URGENT_RE.search(message))


def llm_check(llm: BaseChatModel, message: str) -> bool:
    response = llm.invoke(_LLM_CLASSIFIER_PROMPT.format(message=message))
    return response.content.strip().lower().startswith("y")


def is_urgent(llm: BaseChatModel, message: str) -> bool:
    return keyword_check(message) or llm_check(llm, message)
