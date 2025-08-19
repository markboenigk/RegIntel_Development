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
import numpy as np

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
    """Search for similar documents using vector similarity with real Milvus integration."""
    try:
        # Use specified collection or default
        target_collection = collection_name or DEFAULT_COLLECTION
        
        print(f"🔍 DEBUG: Starting search in collection: {target_collection}")
        print(f"🔍 DEBUG: Query: {query}")
        print(f"🔍 DEBUG: Limit: {top_k}")
        
        # Get query embedding for semantic search
        print(f"🔍 DEBUG: About to get embedding for query: '{query}'")
        query_embedding = await get_embedding(query)
        print(f'🔍 DEBUG: Embedding generated, length: {len(query_embedding) if query_embedding else 0}')
        if not query_embedding:
            print("❌ DEBUG: Failed to generate embedding")
            return []
        print(f"🔍 DEBUG: Embedding successful, proceeding with search")

        # First, try to load the collection if it's not loaded
        print(f"🔍 DEBUG: About to load collection '{target_collection}' if needed...")
        load_success = await load_collection_if_needed(target_collection)
        print(f"🔍 DEBUG: Collection loading result: {load_success}")
        
        if not load_success:
            print(f"❌ DEBUG: Failed to load collection '{target_collection}', trying search anyway...")
        
        # Use search endpoint for vector-based search
        search_url = f"{MILVUS_URI}/v2/vectordb/entities/search"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Use different schemas based on collection type
        if target_collection == "fda_warning_letters":
            # FDA Warning Letters schema
            output_fields = [
                "text_content", "company_name", "letter_date", "chunk_type", 
                "chunk_id", "violations", "required_actions", "systemic_issues",
                "regulatory_consequences", "product_types", "product_categories"
            ]
        else:
            # RSS Feeds schema (default)
            output_fields = [
                "text_content", "article_title", "published_date", "feed_name", 
                "chunk_type", "companies", "products", "regulations", "regulatory_bodies"
            ]
        
        # Convert to float32 array (Zilliz expects this)
        query_embedding_float32 = np.array(query_embedding, dtype=np.float32).flatten().tolist()
        
        search_data = {
            "collectionName": target_collection,
            "data": query_embedding_float32,
            "limit": top_k,
            "outputFields": output_fields,
            "metricType": "COSINE",
            "params": {"nprobe": 10},
            "fieldName": "text_vector"
        }
        
        print(f"🔍 DEBUG: Attempting vector search...")
        print(f"🔍 DEBUG: Search URL: {search_url}")
        print(f"🔍 DEBUG: Search data: {json.dumps(search_data, indent=2)}")

        response = requests.post(search_url, json=search_data, headers=headers)
        print(f"🔍 DEBUG: Milvus response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ DEBUG: Zilliz API error: {response.status_code}")
            print(f"❌ DEBUG: Response text: {response.text}")
            return []
        
        result = response.json()
        pretty_json_string = json.dumps(result, indent=4)
        print(f'🔍 DEBUG: Milvus raw response: {pretty_json_string}')
        
        # Check if this is an error response
        if 'code' in result and result.get('code') != 0:
            print(f"❌ DEBUG: Milvus API returned error: Code {result.get('code')}, Message: {result.get('message')}")
            return []

        sources = []
        if 'data' in result:
            print(f"🔍 DEBUG: Found 'data' field in response with {len(result['data'])} items")
            
            for hit in result['data']:
                try:
                    # Create metadata based on collection schema
                    if target_collection == "fda_warning_letters":
                        # FDA Warning Letters metadata
                        metadata = {
                            "company_name": hit.get('company_name', 'Unknown Company'),
                            "letter_date": hit.get('letter_date', 'Unknown Date'),
                            "chunk_type": hit.get('chunk_type', 'Unknown Type'),
                            "chunk_id": hit.get('chunk_id', 'Unknown Chunk'),
                            "violations": hit.get('violations', []),
                            "required_actions": hit.get('required_actions', []),
                            "systemic_issues": hit.get('systemic_issues', []),
                            "regulatory_consequences": hit.get('regulatory_consequences', []),
                            "product_types": hit.get('product_types', []),
                            "product_categories": hit.get('product_categories', [])
                        }
                    else:
                        # RSS Feeds metadata (default)
                        metadata = {
                            "article_title": hit.get('article_title', 'Unknown Title'),
                            "published_date": hit.get('published_date', 'Unknown Date'),
                            "feed_name": hit.get('feed_name', 'Unknown Feed'),
                            "chunk_type": hit.get('chunk_type', 'Unknown Type'),
                            "companies": hit.get('companies', []),
                            "products": hit.get('products', []),
                            "regulations": hit.get('regulations', []),
                            "regulatory_bodies": hit.get('regulatory_bodies', [])
                        }
                    
                    source_item = {
                        "title": hit.get('article_title', metadata.get('company_name', 'Unknown Title')),
                        "content": hit.get('text_content', ''),
                        "metadata": metadata,
                        "collection": target_collection
                    }
                    sources.append(source_item)
                    print(f"🔍 DEBUG: Added source: {source_item['title']}")
                
                except Exception as e:
                    print(f"❌ DEBUG: Error parsing hit: {e}")
                    continue
            
            # Return all sources up to the limit
            sources = sources[:top_k]
            print(f"🔍 DEBUG: Returning {len(sources)} sources to LLM")
            
        else:
            print(f"❌ DEBUG: No 'data' field found in response")
        
        print(f"🔍 DEBUG: Final sources count: {len(sources)}")
        if sources:
            pretty_json_string = json.dumps(sources, indent=4)
            print('🔍 DEBUG: Final sources:', pretty_json_string)
    
        return sources
        
    except Exception as e:
        print(f"❌ DEBUG: Error in search_similar_documents: {e}")
        import traceback
        traceback.print_exc()
        return []

async def load_collection_if_needed(collection_name: str) -> bool:
    """Load a collection into memory if it's not already loaded."""
    try:
        print(f"🔄 DEBUG: Checking if collection '{collection_name}' needs to be loaded...")
        
        # Check collection load status using the describe endpoint
        describe_url = f"{MILVUS_URI}/v2/vectordb/collections/describe"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        describe_data = {
            "collectionName": collection_name
        }
        
        print(f"🔄 DEBUG: Checking collection status at: {describe_url}")
        response = requests.post(describe_url, json=describe_data, headers=headers)
        print(f"🔄 DEBUG: Describe response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ DEBUG: Failed to check collection status: {response.status_code}")
            print(f"❌ DEBUG: Response text: {response.text}")
            return False
        
        collection_info = response.json()
        print(f"🔄 DEBUG: Collection info response: {json.dumps(collection_info, indent=2)}")
        
        load_state = collection_info.get('data', {}).get('load', 'Unknown')
        print(f"🔄 DEBUG: Collection '{collection_name}' load state: {load_state}")
        
        if load_state == "LoadStateNotLoad":
            print(f"🔄 DEBUG: Loading collection '{collection_name}'...")
            
            # Load the collection using the load endpoint
            load_url = f"{MILVUS_URI}/v2/vectordb/collections/load"
            load_data = {
                "collectionName": collection_name
            }
            
            print(f"🔄 DEBUG: Loading collection at: {load_url}")
            load_response = requests.post(load_url, json=load_data, headers=headers)
            print(f"🔄 DEBUG: Load response status: {load_response.status_code}")
            print(f"🔄 DEBUG: Load response text: {load_response.text}")
            
            if load_response.status_code == 200:
                load_result = load_response.json()
                if load_result.get('code') == 0:
                    print(f"✅ DEBUG: Collection '{collection_name}' loaded successfully")
                    return True
                else:
                    print(f"❌ DEBUG: Collection load failed with code: {load_result.get('code')}")
                    return False
            else:
                print(f"❌ DEBUG: Failed to load collection: {load_response.status_code}")
                return False
        else:
            print(f"✅ DEBUG: Collection '{collection_name}' is already loaded")
            return True
            
    except Exception as e:
        print(f"❌ DEBUG: Error loading collection: {e}")
        return False

async def chat_with_gpt(message: str, conversation_history: List[ChatMessage], sources: Optional[List[Dict[str, Any]]] = None) -> str:
    """Chat with GPT using conversation history and optional RAG sources."""
    if not client:
        return "OpenAI client not available. Please check your API key configuration."
    
    try:
        # Build context from sources if available
        context = ""
        collection_type = "general"
        
        if sources:
            # Determine collection type from first source
            first_source = sources[0]
            collection_type = first_source.get('collection', 'general')
            
            # Build context with collection-specific information
            context = f"\n\nRelevant sources from {collection_type.replace('_', ' ').title()}:\n"
            for i, source in enumerate(sources[:3], 1):
                title = source.get('title', 'Unknown Title')
                content = source.get('content', '')[:200]
                metadata = source.get('metadata', {})
                
                # Add collection-specific details
                if collection_type == "fda_warning_letters":
                    company = metadata.get('company_name', 'Unknown Company')
                    date = metadata.get('letter_date', 'Unknown Date')
                    context += f"{i}. {title} - Company: {company}, Date: {date}\n"
                else:
                    feed = metadata.get('feed_name', 'Unknown Feed')
                    date = metadata.get('published_date', 'Unknown Date')
                    context += f"{i}. {title} - Feed: {feed}, Date: {date}\n"
                
                context += f"   {content}...\n\n"
        
        # Build conversation messages
        messages = []
        for msg in conversation_history[-5:]:  # Keep last 5 messages for context
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current message with context
        current_message = message
        if context:
            current_message = f"{message}\n\n{context}"
        
        messages.append({"role": "user", "content": current_message})
        
        # Add system message for RegIntel context with collection awareness
        if collection_type == "rss_feeds":
            system_content = "You are RegIntel, an AI assistant specialized in regulatory intelligence and FDA compliance. You're currently analyzing RSS feeds and regulatory news. Provide helpful, accurate information based on the sources provided, focusing on industry trends, policy updates, and compliance developments."
        elif collection_type == "fda_warning_letters":
            system_content = "You are RegIntel, an AI assistant specialized in regulatory intelligence and FDA compliance. You're currently analyzing FDA warning letters and inspection reports. Provide helpful, accurate information based on the sources provided, focusing on compliance violations, regulatory requirements, and corrective actions."
        else:
            system_content = "You are RegIntel, an AI assistant specialized in regulatory intelligence and FDA compliance. Provide helpful, accurate information based on the sources provided. If no relevant sources are available, clearly state that you cannot provide specific information on that topic."
        
        system_message = {
            "role": "system", 
            "content": system_content
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