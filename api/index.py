from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from typing import List, Dict, Any, Optional

# Create FastAPI app
app = FastAPI(title="RegIntel API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RegIntel API is running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Simple chat endpoint for testing"""
    try:
        # For now, just echo back a simple response
        response = f"I received your message: '{request.message}'. This is a simplified version for Vercel deployment."
        
        return ChatResponse(
            response=response,
            sources=[{"title": "Test Source", "content": "This is a test source for Vercel deployment"}]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/collections")
async def get_collections():
    """Get available collections"""
    return {
        "collections": [
            {"id": "rss_feeds", "name": "Regulatory News", "description": "RSS feeds from regulatory sources"},
            {"id": "fda_warning_letters", "name": "FDA Warning Letters", "description": "FDA compliance documents"}
        ]
    }

# For Vercel serverless deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 