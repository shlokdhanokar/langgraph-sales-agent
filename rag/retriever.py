"""
AutoStream Agent — RAG Retriever

Loads the local knowledge base JSON, converts it into text chunks,
generates embeddings using Google Generative AI, and stores them in
a FAISS vector store for semantic retrieval.
"""

import json
import os
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# Path to the knowledge base JSON file
KB_PATH = Path(__file__).parent / "knowledge_base.json"


def _load_and_chunk_knowledge_base() -> list[Document]:
    """
    Load the knowledge base JSON and flatten it into retrievable
    text chunks. Each chunk is a self-contained piece of information.

    Returns:
        List of LangChain Document objects with metadata.
    """
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    documents = []

    # --- Chunk each plan into its own document ---
    for plan_name, plan_info in kb.get("plans", {}).items():
        price = plan_info.get("price", "N/A")
        features = ", ".join(plan_info.get("features", []))
        content = (
            f"AutoStream {plan_name.capitalize()} Plan:\n"
            f"Price: {price}\n"
            f"Features: {features}"
        )
        documents.append(Document(
            page_content=content,
            metadata={"source": "knowledge_base", "type": "plan", "plan": plan_name}
        ))

    # --- Chunk each policy into its own document ---
    for policy_name, policy_text in kb.get("policies", {}).items():
        content = f"AutoStream {policy_name.capitalize()} Policy: {policy_text}"
        documents.append(Document(
            page_content=content,
            metadata={"source": "knowledge_base", "type": "policy", "policy": policy_name}
        ))

    # --- Add a combined overview document for general queries ---
    all_plans = []
    for plan_name, plan_info in kb.get("plans", {}).items():
        price = plan_info.get("price", "N/A")
        features = ", ".join(plan_info.get("features", []))
        all_plans.append(f"- {plan_name.capitalize()} Plan: {price} — {features}")

    overview_content = (
        "AutoStream Pricing Overview:\n"
        + "\n".join(all_plans)
        + "\n\nPolicies:\n"
        + "\n".join(
            f"- {name.capitalize()}: {text}"
            for name, text in kb.get("policies", {}).items()
        )
    )
    documents.append(Document(
        page_content=overview_content,
        metadata={"source": "knowledge_base", "type": "overview"}
    ))

    return documents


def build_retriever():
    """
    Build and return a FAISS-based retriever from the knowledge base.

    Uses Google Generative AI embeddings for vector generation.
    Returns a LangChain Retriever object that can be queried with
    natural language.

    Returns:
        A FAISS retriever configured to return top-3 results.
    """
    # Load and chunk the knowledge base
    documents = _load_and_chunk_knowledge_base()

    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # Create FAISS vector store from documents
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Return as retriever with top-3 results
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever


def retrieve_context(query: str, retriever) -> str:
    """
    Query the retriever and format the results into a context string.

    Args:
        query:     The user's question or search query.
        retriever: An initialized LangChain retriever.

    Returns:
        Concatenated text from the top matching documents.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the knowledge base."

    # Combine retrieved documents into a single context string
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Source {i}]\n{doc.page_content}")

    return "\n\n".join(context_parts)
