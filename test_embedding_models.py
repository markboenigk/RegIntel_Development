#!/usr/bin/env python3
"""
Test script to check available embedding models and their dimensions
"""

import os
import asyncio
import openai
from dotenv import load_dotenv

async def test_embedding_models():
    """Test different embedding models to see their actual dimensions"""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("🔍 Testing OpenAI Embedding Models and Dimensions")
    print("=" * 50)
    
    if not api_key:
        print("❌ No OpenAI API key found")
        return False
    
    try:
        client = openai.AsyncOpenAI(api_key=api_key)
        test_text = "FDA regulatory compliance requirements"
        
        # Test different models
        models_to_test = [
            "text-embedding-3-small",
            "text-embedding-3-large", 
            "text-embedding-ada-002"
        ]
        
        for model in models_to_test:
            print(f"\n🧪 Testing model: {model}")
            try:
                response = await client.embeddings.create(
                    model=model,
                    input=test_text
                )
                
                embedding = response.data[0].embedding
                print(f"✅ Success: {len(embedding)} dimensions")
                print(f"   First 5 values: {embedding[:5]}")
                
            except Exception as e:
                print(f"❌ Error with {model}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Embedding Models Test")
    print("=" * 50)
    
    success = asyncio.run(test_embedding_models())
    
    if success:
        print("\n🎉 All models tested successfully!")
    else:
        print("\n❌ Testing failed")
    
    print("\n" + "=" * 50) 