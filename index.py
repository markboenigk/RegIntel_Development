import os
import mmh3
import json
import requests
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# Collection names - configurable via environment variables
FDA_WARNING_LETTERS_COLLECTION = os.getenv("FDA_WARNING_LETTERS_COLLECTION", "fda_warning_letters")
RSS_FEEDS_COLLECTION = os.getenv("RSS_FEEDS_COLLECTION", "rss_feeds")
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", RSS_FEEDS_COLLECTION)

# RAG Configuration
STRICT_RAG_ONLY = os.getenv("STRICT_RAG_ONLY", "true").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "o3")
INITIAL_SEARCH_MULTIPLIER = int(os.getenv("INITIAL_SEARCH_MULTIPLIER", "3"))

# Initialize OpenAI client
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_history: List[ChatMessage] = Field(default=[], description="Conversation history")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    sources: List[Dict[str, Any]] = Field(default=[], description="RAG sources")

class AddDocumentRequest(BaseModel):
    text: str = Field(..., description="Document text to add")
    metadata: str = Field(default="", description="Optional metadata for the document")

# Create FastAPI app
app = FastAPI(
    title="RegIntel RAG API",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Utility functions
async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI."""
    if not client:
        return []
    try:
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

async def search_similar_documents(query: str, collection_name: str = "rss_feeds", top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar documents using vector similarity."""
    try:
        # For now, return mock data for testing
        # TODO: Integrate with actual Milvus client
        return [
            {
                "title": f"Regulatory Document from {collection_name}",
                "content": f"This is a sample regulatory document related to: {query}",
                "metadata": {"collection": collection_name, "source": "mock_data"}
            }
        ]
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

async def chat_with_gpt(message: str, conversation_history: List[ChatMessage], sources: Optional[List[Dict[str, Any]]] = None) -> str:
    """Chat with GPT using conversation history and optional RAG sources."""
    if not client:
        return "OpenAI client not available. Please check your API key configuration."
    
    try:
        # Build context from sources if available
        context = ""
        if sources:
            context = "\n\nRelevant sources:\n" + "\n".join([
                f"- {source.get('title', 'Unknown')}: {source.get('content', '')[:200]}..."
                for source in sources[:3]
            ])
        
        # Build conversation messages
        messages = []
        for msg in conversation_history[-5:]:  # Keep last 5 messages for context
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current message with context
        current_message = message
        if context:
            current_message = f"{message}\n\n{context}"
        
        messages.append({"role": "user", "content": current_message})
        
        # Add system message for RegIntel context
        system_message = {
            "role": "system", 
            "content": "You are RegIntel, an AI assistant specialized in regulatory intelligence and FDA compliance. Provide helpful, accurate information based on the sources provided. If no relevant sources are available, clearly state that you cannot provide specific information on that topic."
        }
        messages.insert(0, system_message)
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error chatting with GPT: {e}")
        return f"I encountered an error while processing your request: {str(e)}"

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
    """Chat endpoint with RAG integration."""
    try:
        # Search for relevant documents
        sources = await search_similar_documents(request.message, DEFAULT_COLLECTION)
        
        # Convert conversation history to ChatMessage objects
        history = [ChatMessage(role=msg.role, content=msg.content) 
                  for msg in request.conversation_history]
        
        # Get AI response
        response = await chat_with_gpt(request.message, history, sources)
        
        return ChatResponse(
            response=response,
            sources=sources
        )
        
    except Exception as e:
        print(f"❌ Internal error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/chat/{collection}", response_model=ChatResponse)
async def chat_with_collection(collection: str, request: ChatRequest):
    """Chat endpoint with RAG integration for a specific collection."""
    try:
        # Search for relevant documents in specified collection
        sources = await search_similar_documents(request.message, collection)
        
        # Convert conversation history to ChatMessage objects
        history = [ChatMessage(role=msg.role, content=msg.content) 
                  for msg in request.conversation_history]
        
        # Get AI response
        response = await chat_with_gpt(request.message, history, sources)
        
        return ChatResponse(
            response=response,
            sources=sources
        )
        
    except Exception as e:
        print(f"❌ Internal error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/collections")
async def get_collections():
    """Get available collections"""
    return {
        "collections": [
            {"id": "rss_feeds", "name": "Regulatory News", "description": "RSS feeds from regulatory sources"},
            {"id": "fda_warning_letters", "name": "FDA Warning Letters", "description": "FDA compliance documents"}
        ]
    }

@app.get("/api/rss-feeds/latest")
async def get_latest_rss_feeds(limit: int = 10):
    """Get latest RSS feeds"""
    return {
        "feeds": [
            {
                "id": f"feed_{i}",
                "title": f"Regulatory Update {i}",
                "content": f"Latest regulatory news and updates from various sources",
                "published_at": "2025-08-18T20:00:00Z",
                "source": "Regulatory News Feed"
            }
            for i in range(1, min(limit + 1, 6))
        ]
    }

@app.get("/api/warning-letters/latest")
async def get_latest_warning_letters(limit: int = 10):
    """Get latest FDA warning letters"""
    return {
        "warning_letters": [
            {
                "id": f"wl_{i}",
                "title": f"FDA Warning Letter {i}",
                "content": f"FDA warning letter regarding compliance issues",
                "issued_date": "2025-08-18",
                "company": f"Company {i}",
                "violations": ["Quality System", "Documentation"]
            }
            for i in range(1, min(limit + 1, 6))
        ]
    }

@app.get("/auth/me")
async def get_current_user():
    """Get current user info (placeholder for now)"""
    return {"user": None, "authenticated": False}

@app.get("/auth/login")
async def login_page():
    """Login page (placeholder for now)"""
    return {"message": "Login functionality coming soon"}

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 