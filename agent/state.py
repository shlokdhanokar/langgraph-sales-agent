"""
AutoStream Agent — State Definition

Defines the shared state schema for the LangGraph state machine.
All nodes read from and write to this state to maintain context
across the conversation flow.
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class LeadData(TypedDict, total=False):
    """Structured data for lead capture."""
    name: str
    email: str
    platform: str


class AgentState(TypedDict):
    """
    Core state for the AutoStream conversational agent.

    Fields:
        messages:       Full conversation history (HumanMessage / AIMessage).
                        Uses LangGraph's add_messages reducer for automatic
                        append-on-update behavior.
        intent:         The classified intent for the current user turn.
                        One of: 'greeting', 'product_inquiry', 'high_intent'.
        lead_data:      Partially or fully collected lead information.
        rag_context:    Retrieved knowledge-base context for the current query.
        lead_captured:  Flag indicating whether mock_lead_capture has been called.
        response:       The agent's latest textual response to the user.
    """
    messages: Annotated[list, add_messages]
    intent: str
    lead_data: LeadData
    rag_context: str
    lead_captured: bool
    response: str
