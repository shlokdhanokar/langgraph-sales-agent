# 🎥 AutoStream Agent — Live Demo Script

Use this script to demonstrate the full capabilities of the LangGraph state machine, RAG retrieval, and tool execution.

## The Setup
Before starting the demo, ensure the script is running:
```bash
python main.py
```
*(Point out the initialization message "Loading knowledge base & building embeddings..." to show that RAG is setting up).*

---

## 🟢 Step 1: Greeting & Intent Classification
*Goal: Show that the agent acts as a conversational interface and accurately detects a "greeting" intent.*

**You type:**
> `hello`

**Expected Agent Reaction:**
It will respond warmly, introducing itself as the AutoStream assistant and offering to help with plans or pricing.

---

## 🔵 Step 2: RAG Retrieval (Product Inquiry)
*Goal: Demonstrate RAG capabilities. The agent must pull exact pricing data from its internal vector database without hallucinating.*

**You type:**
> `What plans do you offer?`

**Expected Agent Reaction:**
It will query the FAISS database and list the exact plans from the JSON:
- Basic Plan ($29/month)
- Pro Plan ($79/month)

*(Explain to the audience: "Notice it didn't make up prices. It pulled this strictly from our local knowledge_base.json file via RAG")*

---

## 🟣 Step 3: High Intent Detection & Memory Initiation
*Goal: Trigger the `high_intent` node and show LangGraph memory tracking missing data fields (Name, Email, Platform).*

**You type:**
> `I want the Pro plan for my YouTube channel`

**Expected Agent Reaction:**
The LLM will extract the Platform ("YouTube") automatically, realize it still needs a Name and Email, and seamlessly ask for your Name.

---

## 🟠 Step 4: Step-by-Step Data Collection
*Goal: Prove that the agent holds context across multiple turns using the `MemorySaver` checkpointer.*

**You type:**
> `shlok dhanokar`

**Expected Agent Reaction:**
The agent remembers you want the Pro plan for YouTube, saves your name, and naturally asks for the final missing piece: your email address.

---

## 🔴 Step 5: Tool Execution & Lead Capture
*Goal: Trigger the backend execution CRM tool. Explain that the tool ONLY fires because all 3 validation conditions are met.*

**You type:**
> `shlokdhanokar663@gmail.com`

**Expected Agent Reaction:**
The terminal will suddenly break character and print a structured backend system block:
```text
==================================================
  [OK] LEAD CAPTURED SUCCESSFULLY
==================================================
  Name:     shlok dhanokar
  Email:    shlokdhanokar663@gmail.com
  Platform: youtube
==================================================
```
Following the system print, the Agent will respond with a final closing message confirming the setup is complete and the team will reach out. 

---

## 🏁 Step 6: Session End
*Goal: Gracefully close the demo and show the session stats.*

**You type:**
> `quit`

**Terminal Output:**
The script will output the total turns completed and the unique LangGraph Thread ID that persisted the memory throughout the conversation.
