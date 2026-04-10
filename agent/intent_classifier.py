"""
AutoStream Agent — Intent Classifier

Hybrid intent classification using keyword rules + LLM fallback.
Classifies user input into exactly 3 categories:
    1. greeting        — casual hellos, hi, hey
    2. product_inquiry — questions about plans, pricing, features, policies
    3. high_intent     — user wants to buy, subscribe, sign up, try a plan
"""

import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


# --- Keyword patterns for rule-based classification ---

# High-intent phrases: user explicitly wants to purchase or sign up
HIGH_INTENT_PATTERNS = [
    r"\bi\s+want\s+to\s+buy\b",
    r"\bi\s+want\s+to\s+try\b",
    r"\bi\s+want\s+(the\s+)?pro\b",
    r"\bi\s+want\s+(the\s+)?basic\b",
    r"\bsign\s+me\s+up\b",
    r"\bsign\s+up\b",
    r"\bsubscribe\b",
    r"\bi('?m|\s+am)\s+interested\s+in\s+(buying|subscribing|getting|the\s+pro|the\s+basic)\b",
    r"\bget\s+started\b",
    r"\bi('?d|\s+would)\s+like\s+to\s+(buy|subscribe|get|try)\b",
    r"\bpurchase\b",
    r"\btake\s+(the\s+)?pro\b",
    r"\btake\s+(the\s+)?basic\b",
    r"\bi\s+want\s+pro\b",
    r"\bi\s+want\s+basic\b",
    r"\blet('?s|\s+us)\s+(do|go|start)\b",
    r"\bcount\s+me\s+in\b",
    r"\bready\s+to\s+(buy|start|subscribe)\b",
]

# Greeting phrases
GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening)|sup|yo|what'?s\s+up)\s*[!.,?]*$",
    r"^(hi|hello|hey)\s+(there|everyone|team)\s*[!.,?]*$",
]

# Product inquiry phrases
INQUIRY_PATTERNS = [
    r"\b(pric|cost|plan|feature|package|tier|subscription|offer)\b",
    r"\b(how\s+much|what\s+do\s+you|tell\s+me\s+about)\b",
    r"\b(refund|support|policy|policies)\b",
    r"\b(compare|difference|vs|versus)\b",
    r"\b(basic|pro)\s+(plan|tier|package)\b",
    r"\bwhat\s+(is|are)\s+(included|available)\b",
]


def _rule_based_classify(user_input: str) -> str | None:
    """
    Attempt to classify intent using regex patterns.

    Returns the intent string if a confident match is found,
    or None if the LLM should handle classification.
    """
    text = user_input.lower().strip()

    # Check high-intent FIRST (more specific patterns)
    for pattern in HIGH_INTENT_PATTERNS:
        if re.search(pattern, text):
            return "high_intent"

    # Check greetings (full-line match for precision)
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, text):
            return "greeting"

    # Check product inquiry
    for pattern in INQUIRY_PATTERNS:
        if re.search(pattern, text):
            return "product_inquiry"

    return None  # Let LLM decide


def _llm_classify(user_input: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Use the LLM as a fallback classifier when rules are inconclusive.

    The LLM is constrained to return ONLY one of the three valid categories.
    """
    system_prompt = """You are an intent classifier for AutoStream, a video editing SaaS company.
Classify the user's message into EXACTLY one of these three categories:

1. "greeting" — The user is saying hello, hi, or any casual greeting.
2. "product_inquiry" — The user is asking about plans, pricing, features, refund policy, or support.
3. "high_intent" — The user explicitly wants to buy, subscribe, sign up, try, or get started with a plan.

IMPORTANT RULES:
- Only classify as "high_intent" if the user EXPLICITLY expresses desire to purchase, subscribe, or sign up.
- Asking a question about pricing is "product_inquiry", NOT "high_intent".
- "Sounds good" or "Interesting" alone is NOT high_intent unless paired with purchase language.
- If the user says something like "I want Pro plan" or "I want to try Pro" — that IS high_intent.

Respond with ONLY the category name. Nothing else. No quotes, no explanation."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User message: {user_input}")
    ]

    response = llm.invoke(messages)
    result = response.content.strip().lower()

    # Validate LLM response — fall back to product_inquiry if unrecognized
    valid_intents = {"greeting", "product_inquiry", "high_intent"}
    if result not in valid_intents:
        # Try to extract a valid intent from the response
        for intent in valid_intents:
            if intent in result:
                return intent
        return "product_inquiry"  # Safe fallback

    return result


def classify_intent(user_input: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Hybrid intent classifier: tries rule-based matching first,
    falls back to LLM for ambiguous inputs.

    Args:
        user_input: The user's message text.
        llm:        Initialized LLM instance for fallback classification.

    Returns:
        One of: 'greeting', 'product_inquiry', 'high_intent'
    """
    # Step 1: Try rule-based classification
    rule_result = _rule_based_classify(user_input)
    if rule_result is not None:
        return rule_result

    # Step 2: Fall back to LLM classification
    return _llm_classify(user_input, llm)
