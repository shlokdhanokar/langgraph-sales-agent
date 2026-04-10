<h1 align="center">🎬 AutoStream Conversational AI Agent</h1>

<p align="center">
  <em>A production-grade, state-machine driven Conversational AI Agent built for automated lead generation and product inquiry handling.</em>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-latest-green.svg" />
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-latest-orange.svg" />
  <img alt="Gemini" src="https://img.shields.io/badge/Google%20Gemini-v2.5_Flash-4285F4.svg" />
  <img alt="FAISS" src="https://img.shields.io/badge/FAISS-Vector_Store-red.svg" />
</p>

---

## 📖 Overview

The **AutoStream Agent** is a cutting-edge conversational AI designed for the fictional SaaS company, AutoStream (Automated Video Editing). Rather than a simple chatbot, it operates as a **decision-making agent** with logic, memory, and robust tool execution guarantees.

### 🌟 Key Capabilities
- **Hybrid Intent Classification**: Combines fast regex rules with LLM-based fallback to accurately categorize user inputs into 3 states: `greeting`, `product_inquiry`, or `high_intent`.
- **RAG-Powered Knowledge Retrieval**: Answers product and pricing questions securely by querying a local JSON knowledge base using FAISS and Google Generative AI embeddings. **Zero hallucinations.**
- **Robust Tool Execution**: Captures qualified leads by executing backend CRM tools *only* when all necessary data points (Name, Email, Platform) are successfully collected and validated.
- **Persistent Multi-Turn Memory**: Driven by an orchestrated LangGraph state machine and `MemorySaver`, allowing the agent to remember context seamlessly across lengthy conversations.

---

## 📁 Project Architecture

```text
autostream-agent/
│
├── main.py                       # Interactive CLI entry point
├── agent/
│   ├── graph.py                  # LangGraph state machine definition
│   ├── state.py                  # Shared state schema (TypedDict)
│   ├── nodes.py                  # 5 core logic nodes
│   ├── tools.py                  # CRM mock_lead_capture tool
│   ├── intent_classifier.py      # Hybrid intent detection logic
│
├── rag/
│   ├── knowledge_base.json       # Product plans & policies data
│   ├── retriever.py              # FAISS embedding & retrieval engine
│
├── utils/
│   ├── helpers.py                # Regex validations & data checks
│
├── requirements.txt              # Standard Python dependencies
└── .env.example                  # Environment variable blueprint
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- A Google API key (Available for free at [Google AI Studio](https://aistudio.google.com/apikey))

### 2. Setup

```bash
# Clone or navigate to the project repository
cd autostream-agent

# Create and activate a virtual environment
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Configure your environment variables
cp .env.example .env
```

Open `.env` and configure your API key:
```env
GOOGLE_API_KEY=your-actual-api-key-here
```

### 3. Run the Agent

```bash
python main.py
```

---

## 💬 Sample Flow & Demo

1. **User:** `"Hi there!"`
   *Agent classifies as `greeting` & responds warmly.*
2. **User:** `"Tell me about your pricing."`
   *Agent classifies as `product_inquiry`, retrieves data from FAISS, and outputs exact pricing.*
3. **User:** `"I want the Pro plan for YouTube."`
   *Agent detects `high_intent`, extracts "YouTube", and prompts for the missing `Name` and `Email`.*
4. **User:** `"My name is John, email is john@example.com."`
   *Agent validates data completeness, executes `mock_lead_capture()`, and finalizes the interaction.*

---

## 🏗️ Deep Dive: Core Technologies (~200 Words)

### Why LangGraph over LangChain?
LangGraph was explicitly chosen over standard LangChain sequential chains because this agent requires **non-linear conditional branching** and strict **stateful multi-turn flow**. LangGraph models the agent as a **cyclical state machine**. Each function acts as a "node", and conditional edges route the conversation dynamically depending on variables like detected intent and data completeness. This granular control allows complex flows—like interrupting a user to ask for missing lead fields—without dropping the context. 

### How State & Memory Work
The agent leverages a carefully structured `TypedDict` (`AgentState`) carrying the message history, classified intent, accumulated lead data, and retrieved RAG context. LangGraph's internal `MemorySaver` checkpointer serializes this state between turns, enabling the agent to remember prior context natively.

### RAG Strategy
Pricing limits and policies are mapped out in `knowledge_base.json`. We flatten this hierarchy into document chunks, embed them using **Google Generative AI**, and load them directly into an in-memory **FAISS vector store**. This ensures maximum data control and zero LLM "hallucination".

---

## 📱 WhatsApp Integration Guide

Scaling this agent to WhatsApp requires an internet-facing server acting as a webhook handler for the **WhatsApp Business API**. 

### Architecture Flow
`User Message ↔ WhatsApp Business API ↔ Flask Webhook ↔ LangGraph Agent`

### 1. Webhook Implementation (Flask)

Deploy a Flask server to receive payloads from WhatsApp:

```python
from flask import Flask, request, jsonify
from agent.graph import build_agent_graph
from langchain_core.messages import HumanMessage
import uuid

app = Flask(__name__)
agent = build_agent_graph()

# In-memory session store (Use Redis for production scaling)
session_store = {}

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    payload = request.json
    try:
        message_data = payload["entry"][0]["changes"][0]["value"]["messages"][0]
        sender_phone = message_data["from"]
        user_text = message_data["text"]["body"]
        
        # Link sender phone number to a consistent LangGraph Thread ID
        if sender_phone not in session_store:
            session_store[sender_phone] = str(uuid.uuid4())
            
        config = {"configurable": {"thread_id": session_store[sender_phone]}}
        
        # Stream text into the agent
        result = agent.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
        agent_response = result.get("response", "Could you rephrase that?")
        
        # Route back to user
        send_whatsapp_message(sender_phone, agent_response)
        
    except KeyError:
        pass # Ignore statuses/non-text updates 
        
    return jsonify({"status": "delivered"}), 200
```

### 2. Message Dispatcher

Send out the final generated response:

```python
import os, requests

def send_whatsapp_message(destination_phone: str, message_body: str):
    url = f"https://graph.facebook.com/v18.0/{os.getenv('WA_PHONE_ID')}/messages"
    headers = {
        "Authorization": f"Bearer {os.getenv('WA_ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": destination_phone,
        "text": {"body": message_body}
    }
    requests.post(url, headers=headers, json=payload)
```

### Deployment Checklist
- **Hosting**: AWS, Google Cloud Run, Railway, or Heroku.
- **SSL**: WhatsApp requires strict `HTTPS` webhooks.
- **Meta Verification**: Add the Webhook URL in your Meta Developer Dashboard and verify it via a custom token challenge validation loop.
- **Concurrency & State**: Move `session_store` keys out of local memory into Redis/Postgres for multi-worker container deployments.

---

### 📄 License & Disclaimer
This is an educational prototype and demonstration of production-grade GenAI architectural patterns.
