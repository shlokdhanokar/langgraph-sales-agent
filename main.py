"""
AutoStream Agent — Main Entry Point

Interactive CLI interface for the AutoStream conversational AI agent.
Manages the conversation loop, user input, and graph invocation with
persistent memory across turns.

Usage:
    python main.py
"""

import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import build_agent_graph


def main():
    """Run the AutoStream agent in interactive CLI mode."""

    # Load environment variables from .env file
    load_dotenv()

    # Validate API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n[ERROR] GOOGLE_API_KEY not found.")
        print("   Please create a .env file with your Google API key.")
        print("   See .env.example for reference.\n")
        sys.exit(1)

    # --- Initialize the agent graph ---
    print("\n[*] Initializing AutoStream Agent...")
    print("   Loading knowledge base & building embeddings...\n")

    try:
        app = build_agent_graph()
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize agent: {e}")
        sys.exit(1)

    # Generate a unique thread ID for this conversation session
    # This enables LangGraph's MemorySaver to persist state across turns
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # --- Welcome message ---
    print("=" * 60)
    print("  >> Welcome to AutoStream AI Assistant")
    print("  Automated Video Editing — Powered by AI")
    print("=" * 60)
    print("  Type your message below. Type 'quit' or 'exit' to end.\n")

    # --- Conversation loop ---
    turn_count = 0

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Handle exit commands
            if user_input.lower() in ("quit", "exit", "bye", "q"):
                print("\nThank you for chatting with AutoStream! Goodbye.\n")
                break

            # Skip empty input
            if not user_input:
                continue

            turn_count += 1

            # Invoke the agent graph with the user's message
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

            # Extract and display the agent's response
            response = result.get("response", "I'm sorry, I couldn't process that.")
            print(f"\nAutoStream: {response}\n")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n[WARNING] Error: {e}")
            print("   Please try again.\n")

    # --- Session summary ---
    print(f"[STATS] Session Stats: {turn_count} turns completed.")
    print(f"[KEY] Thread ID: {thread_id}\n")


if __name__ == "__main__":
    main()
