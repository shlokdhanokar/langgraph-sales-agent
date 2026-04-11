# AutoStream Sales Agent: Demo Video Script & Walkthrough

This document contains a suggested script and prompts for your 2-3 minute screen-recorded demo video. The goal is to quickly and clearly demonstrate the three core specific requirements from the assignment: **Intent Detection, RAG (Product Knowledge), and Tool Execution (Lead Capture).**

---

## 🎬 Tips for the Video
- **Keep it pacing:** 2-3 minutes is short. Don't spend too much time explaining code line-by-line; focus on the *behavior* of the agent in the UI/terminal.
- **Show, Don't Just Tell:** Start the agent in the terminal and interact with it while you talk. 
- **Highlight LangGraph:** Mention that the state management, memory, and transitions between normal chat, RAG, and lead capture are handled natively by LangGraph nodes and conditional edges.

---

## 📝 Demo Prompts

Copy and paste these prompts during your recording to flawlessly demonstrate each capability.

**Prompt 1: The Casual Greeting (Intent: greeting)**
> "hello"
*Goal: Show that the agent can gracefully handle casual chat without trying to hard-sell or search documentation.*

**Prompt 2: The RAG / Product Query (Intent: product_inquiry)**
> "What plans do you offer?"
*Goal: Show the agent using FAISS-powered RAG to fetch specific, factual pricing information from the provided AutoStream JSON knowledge base (avoiding hallucinations).*

**Prompt 3: The Lead Intent (Intent: high_intent entry)**
> "I want the Pro plan for my YouTube channel"
*Goal: Show the intent classifier routing the user to the lead capture node. The agent detects the platform ("YouTube") but asks for the remaining missing details.*

**Prompt 4: Partial Information (Lead Capture multi-turn)**
> "shlok dhanokar"
*Goal: Demonstrate multi-turn state memory. The agent saves the Name, realizes Email is still missing from the required tool schema, and specifically asks for the email address.*

**Prompt 5: Fulfilling the Tool Execution (Lead Capture successful)**
> "shlokdhanokar663@gmail.com"
*Goal: Show the backend executing the CRM lead capture tool, printing the validated payload to the terminal, and the agent thanking the user.*

---

## 🎙️ Video Script (Target: 2.5 Minutes)

### [0:00 - 0:30] Introduction & Architecture
**(Visual: Show your terminal starting up `python main.py`, briefly show the terminal initialization message)**

**Speaker:** 
"Hello! Thank you for the opportunity to present my assignment. Today I'm demonstrating the Conversational Sales Agent I built for the fictional SaaS product, AutoStream. 

This agent is powered by Google Gemini and orchestrated entirely using LangGraph. LangGraph handles the state management, routing the conversation cleanly between three core behaviors based on user intent: casual chat, product FAQs referencing a RAG pipeline, and a structured lead capture state machine."

### [0:30 - 1:15] Demo Pt 1: Casual Chat & RAG
**(Visual: Type/Paste Prompt 1 `hello` into the chat)**

**Speaker:** 
"Let's start with a normal greeting. When I say 'hello', the hybrid intent classifier detects a greeting intent and responds naturally without doing any heavy lifting."

**(Visual: Type/Paste Prompt 2 `What plans do you offer?` into the chat)**

**Speaker:** 
"Now, let's ask a specific product question: *'What plans do you offer?'* Here, the agent detects a product inquiry constraint. It immediately queries an in-memory FAISS vector store to retrieve factual pricing plans from our local JSON knowledge base, ensuring we never hallucinate prices to a customer."

### [1:15 - 2:00] Demo Pt 2: Intent Routing & Memory
**(Visual: Type/Paste Prompt 3 `I want the Pro plan for my YouTube channel` into the chat)**

**Speaker:** 
"Next, I'll indicate buying intent by saying: *'I want the Pro plan for my YouTube channel.'* 

The system instantly flags this as a high-intent user and transitions the LangGraph state into 'Lead Capture' mode. Notice how the agent extracted 'YouTube' as the platform, but now shifts its behavior to ask for my Name and Email."

**(Visual: Type/Paste Prompt 4 `shlok dhanokar` into the chat)**

**Speaker:**
"To demonstrate state memory across turns, I’ll only provide my name: *'shlok dhanokar'.* 

Because of LangGraph's persistent `MemorySaver` checkpointer, the agent actively records the name, evaluates the required tool schema, identifies that the email is still missing, and gracefully asks *only* for the email address."

### [2:00 - 2:40] Demo Pt 3: Tool Execution & Conclusion
**(Visual: Type/Paste Prompt 5 `shlokdhanokar663@gmail.com` into the chat. Highlight the backend terminal log that prints the captured lead)**

**Speaker:**
"Finally, I'll provide the email: *'shlokdhanokar663@gmail.com'*. 

Now that all three required parameters are successfully verified and met, the agent executes the backend lead capture tool. You can see right here in the terminal logs that a structured payload was pushed cleanly to our mock CRM system, and the agent delivers a highly contextual thank-you message.

Thank you again for reviewing my submission. I'm really excited about the opportunity and look forward to your feedback!"
