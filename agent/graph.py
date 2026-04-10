"""
AutoStream Agent — LangGraph State Machine

Assembles the complete agent graph with conditional routing:

    User Input
        ↓
    classify_intent
        ↓ (conditional)
    ├── greeting       → response_generator → END
    ├── product_inquiry→ rag_retriever → response_generator → END
    └── high_intent    → lead_collector
                            ↓ (conditional)
                        ├── incomplete → response_generator → END
                        └── complete   → tool_executor → response_generator → END
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    classify_intent_node,
    rag_retriever_node,
    response_generator_node,
    lead_collector_node,
    tool_executor_node,
)
from rag.retriever import build_retriever
from utils.helpers import is_lead_data_complete


# --- Build the retriever once at module level (shared across invocations) ---
_retriever = None


def _init_retriever():
    """Initialize the RAG retriever (lazy loading)."""
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


# ------------------------------------------------------------------ #
#  Routing Functions (conditional edges)
# ------------------------------------------------------------------ #

def route_by_intent(state: dict) -> str:
    """
    Route from the intent classifier to the appropriate next node.

    Returns:
        Node name to transition to.
    """
    intent = state.get("intent", "greeting")

    if intent == "high_intent":
        return "lead_collector"
    elif intent == "product_inquiry":
        return "rag_retriever"
    else:
        # greeting or unknown → generate a friendly response
        return "response_generator"


def route_after_lead_collection(state: dict) -> str:
    """
    Route after lead collection: if all data is present, execute tool;
    otherwise, generate a response asking for missing fields.

    Returns:
        Node name to transition to.
    """
    lead_data = state.get("lead_data", {})

    if is_lead_data_complete(lead_data):
        return "tool_executor"
    else:
        return "response_generator"


# ------------------------------------------------------------------ #
#  Graph Builder
# ------------------------------------------------------------------ #

def build_agent_graph():
    """
    Construct and compile the LangGraph state machine with
    MemorySaver checkpointer for multi-turn conversation persistence.

    Returns:
        Compiled LangGraph application ready for invocation.
    """
    # Initialize the retriever
    _init_retriever()

    # Create the state graph
    graph = StateGraph(AgentState)

    # --- Add all nodes ---
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("rag_retriever", rag_retriever_node)
    graph.add_node("response_generator", response_generator_node)
    graph.add_node("lead_collector", lead_collector_node)
    graph.add_node("tool_executor", tool_executor_node)

    # --- Set entry point ---
    graph.set_entry_point("classify_intent")

    # --- Add conditional edges ---

    # From intent classifier → route by detected intent
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "lead_collector": "lead_collector",
            "rag_retriever": "rag_retriever",
            "response_generator": "response_generator",
        }
    )

    # From RAG retriever → always goes to response generator
    graph.add_edge("rag_retriever", "response_generator")

    # From lead collector → conditional on data completeness
    graph.add_conditional_edges(
        "lead_collector",
        route_after_lead_collection,
        {
            "tool_executor": "tool_executor",
            "response_generator": "response_generator",
        }
    )

    # From tool executor → generate confirmation response
    graph.add_edge("tool_executor", "response_generator")

    # Response generator is always the final node → END
    graph.add_edge("response_generator", END)

    # --- Compile with memory checkpointer ---
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    return app
