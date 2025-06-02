from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from final import get_compliance_response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional


app = FastAPI(
    title="Rhythm Compliance Support Chatbot API",
    description="An AI-powered customer support system for Rhythm session rules.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity; adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    
@app.get("/health")
def health_check():
    return {"status": "ok"}



@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        answer = get_compliance_response(query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")