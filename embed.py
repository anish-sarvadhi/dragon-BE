import json
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()


JSON_PATH = "session_rules_full.json"
FAISS_DIR = "rules_index"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Load the JSON file
with open("session_rules_full.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# Extract the rules list
rules = json_data["Session Rules for All Agencies + Rhythm agency's Exceptions"]

# Convert each rule into a LangChain Document
doc_objects = []
for rule in rules:
    combined_content = f"""
Rule Content: {rule['rule_content']}
Provider Resolution: {rule.get('provider_resolution', 'No resolution provided.')}
Admin Resolution: {rule.get('admin_resolution', 'No resolution provided.')}
""".strip()

    doc_objects.append(
        Document(
            page_content=combined_content,
            metadata={
                "section": rule["section"],
                "rule_number": rule["rule_number"],
                "rule_title": rule["rule_title"],
                "provider_resolution": rule.get("provider_resolution", "No resolution provided."),
                "admin_resolution": rule.get("admin_resolution", "No resolution provided.")
            }
        )
    )

# for doc in doc_objects[:20]:
#     print(doc)

try:
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )
    print("✅ OpenAI Embedding model initialized.")
except Exception as e:
    print(f"❌ Error initializing OpenAI embeddings: {e}")
    exit(1)


# try:
#     vectorstore = FAISS.from_documents(doc_objects, embedding_model)
#     vectorstore.save_local(FAISS_DIR)
#     print(f"✅ FAISS index created and saved at '{FAISS_DIR}'.")
# except Exception as e:
#     print(f"❌ Error creating or saving FAISS vector store: {e}")
#     exit(1)

vectorstore = FAISS.from_documents(doc_objects, embedding_model)
query = "EVV limit exceeded"
results = vectorstore.similarity_search(query, k=3)


for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Section: {doc.metadata['section']}")
    print(f"Rule Number: {doc.metadata['rule_number']}")
    print(f"Rule Title: {doc.metadata['rule_title']}")
    print(f"Provider Resolution: {doc.metadata['provider_resolution']}")
    print(f"Admin Resolution: {doc.metadata['admin_resolution']}")
    print(f"Content:\n{doc.page_content}")

