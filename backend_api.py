from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from final import get_compliance_response  # Import your chatbot logic function
from fastapi.middleware.cors import CORSMiddleware

# FastAPI App Initialization
app = FastAPI(
    title="Rhythm Compliance Support Chatbot API",
    description="An AI-powered customer support system for Rhythm session rules.",
    version="1.0.0"
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None  # Optional thread_id for multi-turn conversations

class QueryResponse(BaseModel):
    answer: str
    thread_id: str  # Always returned for continuity

# Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Query Endpoint
@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Pass both query and thread_id to the chatbot logic
        result = get_compliance_response(query, request.thread_id)
        return result  # Must include both answer and thread_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
