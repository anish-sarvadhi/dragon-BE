import os
import uuid
import dotenv
from typing import Annotated, Optional, TypedDict
from operator import add
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import json

# Load environment variables
dotenv.load_dotenv()
FAISS_DIR = "rules_index_unified"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# System Prompt: Updated (No HCBS/Non-HCBS)
system_prompt = SystemMessage(
    """
    You are an empathetic, knowledgeable, and friendly customer support agent for Rhythm session rules.  
Your primary role is to assist users—including Providers (Teachers), Admins, and support staff—by answering their questions, resolving scenarios, and explaining error messages related to session compliance rules.

Your goal is to provide concise, role-specific, and easy-to-understand answers that help the user quickly resolve their issue. Focus on clarity, brevity, and actionable steps. Avoid robotic, technical, or overly formal language—respond like a helpful human agent.

---

Instructions for Handling Queries:

1 **Clarifying Questions Before Answering**:
   - Always begin by confirming key details before answering:
     - "Could you please tell me your role? Are you a Provider (Teacher) or an Admin?"
   - Wait for the user's response before generating your answer.
   - Once the user’s role is known, proceed with the final answer.

---

2 **Answer Structure (Follow This Format Exactly)**:

When answering the user's query:

- Start your response with:  
  **“Here’s what’s going on:”**

- In **1–2 short sentences**, explain why the issue, error, or scenario is occurring, using **plain, simple language**.

- Mention the **relevant rule number and title only if it adds clarity**.  
  - Example: "This is based on Rule 5.22: Session Time Limits."
  - Do **not** include the full rule content unless the user explicitly asks for it.

- Provide **clear, role-specific next steps** based on the user’s role:  
  - If the user is a **Provider (Teacher)**, share **only Provider-specific actions**.  
  - If the user is an **Admin**, share **only Admin-specific checks**.  
  - Avoid giving steps for both roles—**focus only on the user’s role** to keep the response concise.

- Use **bullet points** for next steps if it improves clarity.

---

3 **Types of Queries to Handle**:
   - Direct questions: (e.g., "How do I add a session?")
   - Error messages: (e.g., "You cannot add a session at this time.")
   - Scenario-based queries: (e.g., "A provider is not able to add a session after approval.")
   - Multi-rule scenarios: (e.g., "What happens if a session overlaps and the provider is not assigned?")

For each, follow the structured format:  
**Here’s what’s going on:** → **Why it happens** → **Role-specific next steps**.

---

4 **Handling Vague or Incomplete Queries**:
   - If a query lacks information, **gently ask follow-up questions** to clarify details (e.g., role, specific issue).
   - While waiting for the user’s response, offer **possible causes or suggestions** based on the information available, but make it clear that more details are needed for a precise answer.

---

5 **Tone and Style**:
   - Be warm, patient, and empathetic—like a real customer support agent, not a robot.
   - Use natural, conversational language.
   - Avoid overly technical terms or jargon unless necessary.
   - Keep responses **short, focused, and to the point**—users want quick, actionable help, not long explanations.

---

6 **Rule Content Handling**:
   - Do not include full rule content by default.
   - Only provide full rule content **if the user explicitly asks for it**.
   - When including a rule, place it **after** your explanation and next steps, never before.
   - Format any quoted rule content clearly for easy reading.

Your ultimate goal is to:
    Quickly identify the issue.  
    Explain it in simple terms.  
    Provide clear, role-specific steps to resolve it.  
    Keep the conversation human, helpful, and easy to follow.
    """
)

# State Schema (context removed)
class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    role: Optional[str]
    next: Optional[str]

# Extract Role (Only)
def extract_role_via_llm(user_input: str):
    extraction_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("""
        Extract the user's role (Provider/Teacher or Admin) from the following message. Return JSON:
        {"role": "..."}
        """),
        HumanMessage(user_input),
    ])
    extraction_chain = extraction_prompt | llm | StrOutputParser()
    result = extraction_chain.invoke({"messages": [HumanMessage(content=user_input)]})

    try:
        result_json = json.loads(result)
        return result_json.get("role")
    except json.JSONDecodeError:
        return None

# Format Rules
def format_rules(docs):
    if not docs:
        return "No relevant rules found."
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        content = doc.page_content
        provider_res = meta.get("provider_resolution", "No resolution provided.")
        admin_res = meta.get("admin_resolution", "No resolution provided.")
        category = meta.get("category", "N/A")
        formatted.append(f"""
{i}. Rule Number: {meta.get('rule_number', 'N/A')}
   Title: {meta.get('rule_title', 'N/A')}
   Section: {meta.get('section', 'N/A')}
   Category: {category}
   Content: {content.strip()}
   Provider Resolution: {provider_res}
   Admin Resolution: {admin_res}
""")
    return "\n".join(formatted)

# Retrieve Rules (No context filtering)
def retrieve_rules(query, k=5):
    results = vectorstore.similarity_search(query, k=k)
    return format_rules(results)

# Graph Nodes
def extract_role(state: ConversationState):
    last_message = state["messages"][-1].content if state["messages"] else ""
    role = extract_role_via_llm(last_message)
    state["role"] = role or state.get("role")
    return state

def check_clarifications(state: ConversationState):
    if not state.get("role"):
        state["next"] = "ask_clarifications"
    else:
        state["next"] = "answer_query"
    return state

def route_next_step(state: ConversationState):
    return state.get("next", "answer_query")

def ask_clarifications(state: ConversationState):
    clarification_message = "Could you please tell me your role? Are you a Provider (Teacher) or an Admin?"
    state["messages"].append(AIMessage(content=clarification_message))
    return state

def answer_query(state: ConversationState):
    last_user_message = state["messages"][-1].content if state["messages"] else ""
    rules = retrieve_rules(last_user_message)
    prompt_with_context = ChatPromptTemplate.from_messages([
        system_prompt,
        ("system", f"User Role: {state['role']}"),
        ("system", f"Relevant Rules:\n{rules}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt_with_context | llm | StrOutputParser()
    response = chain.invoke({"messages": state["messages"]})
    state["messages"].append(AIMessage(content=response))
    return state

# LangGraph Setup
builder = StateGraph(ConversationState)
builder.add_node("extract_role", extract_role)
builder.add_node("check_clarifications", check_clarifications)
builder.add_node("ask_clarifications", ask_clarifications)
builder.add_node("answer_query", answer_query)

builder.add_edge("extract_role", "check_clarifications")
builder.add_conditional_edges("check_clarifications", route_next_step)
builder.add_edge("ask_clarifications", "extract_role")
builder.set_entry_point("extract_role")
builder.set_finish_point("answer_query")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)





def get_compliance_response(user_query: str, thread_id: Optional[str] = None) -> dict:
    import uuid  # Ensure uuid is imported if not already

    # If no thread_id provided, generate a new one
    if not thread_id:
        thread_id = str(uuid.uuid4())

    # Set up graph configuration
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": [HumanMessage(content=user_query)], "role": None, "next": None}

    # Invoke the graph
    state_update = graph.invoke({"messages": state["messages"]}, config=config)
    state.update(state_update)

    # Get the chatbot's final answer
    final_answer = state["messages"][-1].content

    # Return both answer and thread_id
    return {
        "answer": final_answer,
        "thread_id": thread_id
    }


# Main Loop
# if __name__ == "__main__":
#     print("📌 Rhythm Compliance Chatbot Ready!")
#     print("Type your query below (type 'exit' to quit).")

#     thread_id = str(uuid.uuid4())
#     config = {"configurable": {"thread_id": thread_id}}
#     state = {"messages": [], "role": None, "next": None}

#     try:
#         while True:
#             user_input = input("\nUser Query: ").strip()
#             if user_input.lower() in ["exit", "quit"]:
#                 print("👋 Goodbye!")
#                 break
#             if not user_input:
#                 print("⚠️ Please enter a query or type 'exit' to quit.")
#                 continue

#             state["messages"].append(HumanMessage(content=user_input))
#             state_update = graph.invoke({"messages": state["messages"]}, config=config)
#             state.update(state_update)

#             last_ai_message = state["messages"][-1].content
#             print("\nFinal Answer:\n", last_ai_message)

#     except KeyboardInterrupt:
#         print("\n👋 Chatbot stopped. Goodbye!")
