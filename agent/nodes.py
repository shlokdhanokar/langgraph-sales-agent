"""
AutoStream Agent — Graph Nodes

Each function in this module is a node in the LangGraph state machine.
Nodes read from and write to the shared AgentState to drive the
conversation forward.

Nodes:
    1. classify_intent_node   — Detects user intent (greeting / inquiry / high_intent)
    2. rag_retriever_node     — Fetches relevant context from knowledge base
    3. response_generator_node— Generates the agent's reply using LLM + context
    4. lead_collector_node    — Extracts and accumulates lead data from messages
    5. tool_executor_node     — Validates and calls mock_lead_capture
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.intent_classifier import classify_intent
from agent.tools import mock_lead_capture
from rag.retriever import retrieve_context
from utils.helpers import is_lead_data_complete, get_missing_fields


def _get_llm() -> ChatGoogleGenerativeAI:
    """Get a shared LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )


# ------------------------------------------------------------------ #
#  NODE 1: Intent Classifier
# ------------------------------------------------------------------ #

def classify_intent_node(state: dict) -> dict:
    """
    Classify the latest user message into one of three intents.

    If the agent is already in the middle of collecting lead data
    (high_intent detected previously and data incomplete), we keep
    the intent as 'high_intent' to continue the collection flow.
    """
    messages = state.get("messages", [])
    lead_data = state.get("lead_data", {})
    lead_captured = state.get("lead_captured", False)

    # Get the latest user message
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"intent": "greeting"}

    # If we're already collecting lead data and haven't captured yet,
    # stay in high_intent mode to continue collection
    if (state.get("intent") == "high_intent"
            and not lead_captured
            and not is_lead_data_complete(lead_data)):
        return {"intent": "high_intent"}

    # Classify fresh intent
    llm = _get_llm()
    intent = classify_intent(user_message, llm)

    return {"intent": intent}


# ------------------------------------------------------------------ #
#  NODE 2: RAG Retriever
# ------------------------------------------------------------------ #

def rag_retriever_node(state: dict) -> dict:
    """
    Retrieve relevant context from the knowledge base for
    product/pricing inquiries.
    """
    messages = state.get("messages", [])

    # Get the latest user message
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"rag_context": ""}

    # Import here to use the pre-built retriever
    from agent.graph import _retriever
    context = retrieve_context(user_message, _retriever)

    return {"rag_context": context}


# ------------------------------------------------------------------ #
#  NODE 3: Response Generator
# ------------------------------------------------------------------ #

def response_generator_node(state: dict) -> dict:
    """
    Generate the agent's response using the LLM, incorporating:
    - Conversation history (memory)
    - RAG context (if product inquiry)
    - Lead collection status (if high intent)
    """
    messages = state.get("messages", [])
    intent = state.get("intent", "greeting")
    rag_context = state.get("rag_context", "")
    lead_data = state.get("lead_data", {})
    lead_captured = state.get("lead_captured", False)

    llm = _get_llm()

    # Build the system prompt based on current state
    system_prompt = _build_system_prompt(intent, rag_context, lead_data, lead_captured)

    # Construct messages for the LLM
    llm_messages = [SystemMessage(content=system_prompt)]

    # Add conversation history (last 10 messages for context window management)
    for msg in messages[-10:]:
        llm_messages.append(msg)

    # Generate response
    response = llm.invoke(llm_messages)
    response_text = response.content

    return {
        "response": response_text,
        "messages": [AIMessage(content=response_text)]
    }


def _build_system_prompt(intent: str, rag_context: str, lead_data: dict, lead_captured: bool) -> str:
    """Build a dynamic system prompt based on the current conversation state."""

    base = (
        "You are the AutoStream AI assistant — a helpful, professional sales agent "
        "for AutoStream, an Automated Video Editing SaaS platform. "
        "Be concise, friendly, and helpful. Never make up information about plans or pricing."
    )

    if lead_captured:
        return (
            f"{base}\n\n"
            "The lead has been successfully captured. Thank the user warmly, "
            "confirm their details, and let them know the team will reach out soon. "
            "Be enthusiastic and professional."
        )

    if intent == "greeting":
        return (
            f"{base}\n\n"
            "The user has greeted you. Respond warmly and introduce yourself as the "
            "AutoStream assistant. Briefly mention you can help with plan information "
            "or getting started. Keep it short and inviting."
        )

    elif intent == "product_inquiry":
        return (
            f"{base}\n\n"
            "The user is asking about AutoStream's products, plans, pricing, or policies. "
            "Use ONLY the following knowledge base context to answer. "
            "Do NOT hallucinate or invent information.\n\n"
            f"--- KNOWLEDGE BASE CONTEXT ---\n{rag_context}\n--- END CONTEXT ---\n\n"
            "Present the information clearly and offer to help them get started if interested."
        )

    elif intent == "high_intent":
        missing = get_missing_fields(lead_data)
        collected = {k: v for k, v in lead_data.items() if v and str(v).strip()}

        if missing:
            collected_str = ", ".join(f"{k}: {v}" for k, v in collected.items()) if collected else "none yet"
            missing_str = ", ".join(missing)
            return (
                f"{base}\n\n"
                f"The user wants to sign up for AutoStream! You need to collect their details.\n"
                f"Already collected: {collected_str}\n"
                f"Still needed: {missing_str}\n\n"
                f"Ask for the NEXT missing field naturally. Only ask for ONE field at a time. "
                f"Be conversational and encouraging. Do NOT ask for fields already collected."
            )
        else:
            return (
                f"{base}\n\n"
                "All lead data has been collected. The lead capture tool has been executed. "
                "Confirm the signup and thank the user."
            )

    return base


# ------------------------------------------------------------------ #
#  NODE 4: Lead Collector
# ------------------------------------------------------------------ #

def lead_collector_node(state: dict) -> dict:
    """
    Extract lead information (name, email, platform) from the
    latest user message and accumulate it in state.

    Uses LLM to intelligently extract structured data from
    free-form user responses.
    """
    messages = state.get("messages", [])
    lead_data = dict(state.get("lead_data", {}))

    # Get the latest user message
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"lead_data": lead_data}

    # Check if this is the first high-intent message — may contain platform info
    # e.g., "I want Pro plan for YouTube"
    llm = _get_llm()

    # Build extraction prompt
    missing = get_missing_fields(lead_data)
    if not missing:
        return {"lead_data": lead_data}

    extraction_prompt = f"""Extract lead information from the user's message.

Currently missing fields: {', '.join(missing)}
Already collected: {lead_data}

User message: "{user_message}"

For each missing field, extract the value if present in the message:
- name: The user's full name (first name is acceptable)
- email: A valid email address
- platform: A video platform (e.g., YouTube, TikTok, Instagram, Vimeo, etc.)

IMPORTANT:
- Only extract fields that are CLEARLY present in the message.
- Do NOT guess or make up values.
- If a field is not in the message, leave it empty.

Respond in this EXACT format (one field per line, use empty string if not found):
name: <value or empty>
email: <value or empty>
platform: <value or empty>"""

    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    result = response.content.strip()

    # Parse the LLM's structured response
    for line in result.split("\n"):
        line = line.strip()
        if line.startswith("name:"):
            value = line[5:].strip().strip('"').strip("'")
            if value and value.lower() not in ("empty", "not found", "n/a", "none", ""):
                lead_data["name"] = value
        elif line.startswith("email:"):
            value = line[6:].strip().strip('"').strip("'")
            if value and value.lower() not in ("empty", "not found", "n/a", "none", "") and "@" in value:
                lead_data["email"] = value
        elif line.startswith("platform:"):
            value = line[9:].strip().strip('"').strip("'")
            if value and value.lower() not in ("empty", "not found", "n/a", "none", ""):
                lead_data["platform"] = value

    return {"lead_data": lead_data}


# ------------------------------------------------------------------ #
#  NODE 5: Tool Executor
# ------------------------------------------------------------------ #

def tool_executor_node(state: dict) -> dict:
    """
    Execute the mock_lead_capture tool ONLY when all data is collected.

    This node is reached only after lead_collector confirms completeness.
    Validates all inputs before calling the tool.
    """
    lead_data = state.get("lead_data", {})

    # Final safety check — never call without complete data
    if not is_lead_data_complete(lead_data):
        return {"lead_captured": False}

    try:
        result = mock_lead_capture(
            name=lead_data["name"],
            email=lead_data["email"],
            platform=lead_data["platform"]
        )
        return {
            "lead_captured": True,
            "response": result
        }
    except ValueError as e:
        return {
            "lead_captured": False,
            "response": f"Validation error: {str(e)}"
        }
