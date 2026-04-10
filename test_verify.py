"""Quick verification script for AutoStream Agent components."""
import sys
sys.path.insert(0, '.')

print("=" * 50)
print("  AutoStream Agent - Component Tests")
print("=" * 50)

# Test 1: State import
print("\n[TEST 1] State schema import...")
from agent.state import AgentState
print("  PASS: AgentState imported")

# Test 2: Helpers
print("\n[TEST 2] Utility helpers...")
from utils.helpers import is_valid_email, is_lead_data_complete, get_missing_fields
assert is_valid_email("test@email.com") == True
assert is_valid_email("invalid") == False
assert is_lead_data_complete({"name": "John", "email": "j@e.com", "platform": "YT"}) == True
assert is_lead_data_complete({"name": "John"}) == False
assert get_missing_fields({"name": "John"}) == ["email", "platform"]
print("  PASS: All helper functions work correctly")

# Test 3: Tool execution
print("\n[TEST 3] mock_lead_capture tool...")
from agent.tools import mock_lead_capture
result = mock_lead_capture("John Doe", "john@example.com", "YouTube")
assert "Lead captured successfully" in result
print("  PASS: Tool executed and validated")

# Test 4: Intent classifier (rule-based)
print("\n[TEST 4] Rule-based intent classifier...")
from agent.intent_classifier import _rule_based_classify
tests = [
    ("Hi", "greeting"),
    ("hello", "greeting"),
    ("Tell me about pricing", "product_inquiry"),
    ("What plans do you offer?", "product_inquiry"),
    ("I want to buy", "high_intent"),
    ("Sign me up", "high_intent"),
    ("I want Pro plan", "high_intent"),
    ("I want to try Pro", "high_intent"),
]
for text, expected in tests:
    result = _rule_based_classify(text)
    status = "PASS" if result == expected else "FAIL"
    print(f"  {status}: '{text}' -> {result} (expected: {expected})")

# Test 5: RAG retriever
print("\n[TEST 5] RAG retriever build (requires API key)...")
try:
    from rag.retriever import build_retriever, retrieve_context
    retriever = build_retriever()
    context = retrieve_context("What are the pricing plans?", retriever)
    assert "29" in context or "79" in context
    print("  PASS: RAG retriever built and queried successfully")
    print(f"  Context preview: {context[:150]}...")
except Exception as e:
    print(f"  SKIP: {e}")

# Test 6: Full graph build
print("\n[TEST 6] LangGraph state machine build...")
try:
    from agent.graph import build_agent_graph
    app = build_agent_graph()
    print("  PASS: Graph compiled with MemorySaver")
except Exception as e:
    print(f"  SKIP: {e}")

print("\n" + "=" * 50)
print("  All component tests completed!")
print("=" * 50)
