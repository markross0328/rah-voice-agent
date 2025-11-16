from typing import Literal

Intent = Literal["callout", "eta_query", "cancel_or_resched", "smalltalk", "unknown"]

def classify_intent(text: str) -> Intent:
    # TODO: call a small LLM or use regex-based bootstrap
    return "unknown"
