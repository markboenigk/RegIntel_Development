#!/usr/bin/env python3
"""
Test script to verify OpenAI embedding functionality
"""

import os
import asyncio
import openai
from dotenv import load_dotenv

async def test_embedding():
    """Test OpenAI embedding generation"""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("🔍 Testing OpenAI Embedding Generation")
    print("=" * 40)
    print(f"API Key: {api_key[:20]}..." if api_key else "None")
    
    if not api_key:
        print("❌ No OpenAI API key found")
        return False
    
    try:
        # Test 1: Using AsyncOpenAI client (like in your app)
        print("\n🧪 Test 1: Using AsyncOpenAI client...")
        
        client = openai.AsyncOpenAI(api_key=api_key)
        print(f"Client type: {type(client)}")
        print(f"Client methods: {[m for m in dir(client) if not m.startswith('_')]}")
        
        # Test embedding generation
        test_text = "This is a test query for regulatory compliance"
        print(f"Test text: {test_text}")
        
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input=test_text
        )
        
        embedding = response.data[0].embedding
        print(f"✅ Embedding generated successfully!")
        print(f"   Length: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Last 5 values: {embedding[-5:]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with AsyncOpenAI client: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_sync_client():
    """Test with sync OpenAI client as backup"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("\n🧪 Test 2: Using sync OpenAI client...")
    
    try:
        # Test with sync client
        openai.api_key = api_key
        
        test_text = "This is a test query for regulatory compliance"
        response = openai.Embedding.create(
            model="text-embedding-3-large",
            input=test_text
        )
        
        embedding = response['data'][0]['embedding']
        print(f"✅ Sync client embedding generated successfully!")
        print(f"   Length: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with sync client: {e}")
        return False

if __name__ == "__main__":
    print("🔍 OpenAI Embedding Test")
    print("=" * 40)
    
    # Run async test
    async_result = asyncio.run(test_embedding())
    
    if not async_result:
        # Try sync client as fallback
        sync_result = asyncio.run(test_sync_client())
        
        if sync_result:
            print("\n⚠️ Async client failed but sync client works")
            print("   This suggests a client version compatibility issue")
        else:
            print("\n❌ Both async and sync clients failed")
    
    print("\n" + "=" * 40) 