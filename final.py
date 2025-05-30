from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()
FAISS_DIR = "rules_index"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)
reasoning_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)


system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are an empathetic, knowledgeable, and friendly customer support agent for Rhythm session rules. 
    Your primary role is to assist users—including providers (teachers), admins, and support staff—by answering their questions, resolving scenarios, and explaining error messages related to session compliance rules.
    Your goal is to provide clear, actionable, and human-like guidance in every response.
    Instructions for Handling Queries:
    1. Adapt Your Tone and Style Based on the Query Type:
        For direct questions:
            Provide a clear and concise answer.
            Explain the reasoning in simple, non-technical language.
            Offer practical next-step guidance.
        For scenario-based queries:
            Break down the situation into logical steps.
            Explain the logic behind applicable rules.
            Offer step-by-step guidance tailored to the user’s situation.
        For multi-rule or multi-condition cases:
            Clearly explain how different rules interact.
            Clarify which rule(s) take precedence.
            Provide structured guidance for resolution.
        For error messages:
            Express empathy (e.g., "I understand this can be frustrating...").
            Explain why the error occurred in clear terms.
            Provide specific resolution steps for both the Provider (Teacher) and Admin based on the rules.
        For vague or incomplete queries:
            Politely ask clarifying questions to gather more details.
            If possible, provide partial guidance based on available information.
            Encourage the user to share more context for a more accurate solution.
        For queries unrelated to the Rhythm session rules or outside the support scope:
            Respond politely and clearly with:
            "Sorry, I don’t have information on this topic, and I’m unable to assist you with this query."
    2. Enhance Responses Beyond the Rules:
        While your primary knowledge base is the Rhythm session rules, you are also expected to use common sense, attention to detail, and general customer support best practices to guide users more effectively. 
        Enhance your explanations with additional, reasonable insights, practical tips, and real-world context that may not be explicitly stated in the rules but can assist users in understanding the situation better.
    3. Structure of Every Response (Mandatory):
        Each response must include the following three core components without exception:
            A clear explanation of why the issue, question, or error is happening.
            Details of the rule(s) involved, including:
                Rule Number (e.g., 5.22)
                Rule Title
                Rule Section (if applicable)
                Rule Content (copied verbatim from the rules, formatted clearly for easy reading)
            Resolution Steps for Both Roles:
                For Providers (Teachers): What the provider should do to resolve the issue.
                For Admins: What the admin should do to assist or resolve the issue.
    4. Formatting and Style Requirements:
        Always start your response with the phrase:
            "As per the rule(s), here is your answer:"
        Write in a friendly, conversational tone, as if you are a real human customer support agent.
        Use simple, reassuring language; avoid technical jargon or robotic phrasing.
        Present information in a clear, structured, and easy-to-read format.
        When including rule content, use bullet points, numbering, or indentation for clarity.
        Dynamically adapt your tone based on the user's situation—be warm, patient, and proactive.
    """
)
human_prompt = HumanMessagePromptTemplate.from_template("""
User Query:
{query}

Relevant Rules:
{rules}
""")

# === Chain ===
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
output_parser = StrOutputParser()
chain = chat_prompt | reasoning_llm | output_parser

# === Format Rules Function (Includes Resolution Steps) ===
def format_rules(docs):
    if not docs:
        return "No relevant rules found."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        content = doc.page_content
        provider_res = meta.get("provider_resolution", "No resolution provided.")
        admin_res = meta.get("admin_resolution", "No resolution provided.")

        formatted.append(f"""
{i}. Rule Number: {meta.get('rule_number', 'N/A')}
   Title: {meta.get('rule_title', 'N/A')}
   Section: {meta.get('section', 'N/A')}
   Content: {content.strip()}
   Provider Resolution: {provider_res}
   Admin Resolution: {admin_res}
""")
    return "\n".join(formatted)

# === Final Reasoning Function ===
def generate_final_answer(query, k=5):
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    formatted_rules = format_rules(retrieved_docs)
    final_answer = chain.invoke({"query": query, "rules": formatted_rules})
    return final_answer

# === Interactive Testing ===
if __name__ == "__main__":
    print("📌 Rhythm Compliance Chatbot Ready. Type your query below (type 'exit' to quit).")
    while True:
        user_query = input("\nUser Query: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        if not user_query:
            print("⚠️ Please enter a query or type 'exit' to quit.")
            continue
        try:
            answer = generate_final_answer(user_query)
            print("\nFinal Answer:\n", answer)
        except Exception as e:
            print("⚠️ Error:", str(e))
