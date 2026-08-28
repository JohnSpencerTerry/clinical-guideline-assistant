"""Patient-specific scope detection: general guideline question vs. individualized advice.

Deliberately tuned to not over-trigger on conversational-but-general phrasing.
"""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

_PROMPT = """Classify the following message as one of two categories:

- "general": asks what published clinical guidelines say (may be phrased conversationally, e.g. \
"if I have CKD, does that change treatment?" is still general — it's asking about a population/scenario, \
not requesting advice for a specific named individual's actual situation).
- "patient_specific": asks for individualized medical advice about a specific real person (the speaker, \
a named family member, "my" situation with personal details), e.g. "should I stop taking my metformin?" \
or "what should my dad do about his diagnosis?".

Answer with exactly one word: "general" or "patient_specific".

Message: {message}
"""


def classify_scope(llm: BaseChatModel, message: str) -> Literal["general", "patient_specific"]:
    response = llm.invoke(_PROMPT.format(message=message))
    answer = response.content.strip().lower()
    return "patient_specific" if "patient_specific" in answer else "general"
