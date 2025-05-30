from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from final import generate_final_answer
from fastapi.middleware.cors import CORSMiddleware


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

class QueryResponse(BaseModel):
    answer: str
    
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        final_answer = generate_final_answer(query)
        return {"answer": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))