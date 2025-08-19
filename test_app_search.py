#!/usr/bin/env python3
"""
Test script that exactly mimics the app's search process
"""

import os
import asyncio
import requests
import json
import numpy as np
import openai
from dotenv import load_dotenv

async def test_app_search_process():
    """Test the exact search process your app uses"""
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_token = os.getenv("MILVUS_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    print("🔍 Testing App Search Process Step by Step")
    print("=" * 50)
    
    if not all([milvus_uri, milvus_token, openai_api_key]):
        print("❌ Missing required credentials")
        return False
    
    try:
        # Step 1: Generate embedding (exactly like your app)
        print("\n🧪 Step 1: Generating embedding...")
        
        client = openai.AsyncOpenAI(api_key=openai_api_key)
        test_query = "FDA regulatory compliance requirements"
        
        print(f"Query: {test_query}")
        
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=test_query
        )
        
        query_embedding = response.data[0].embedding
        print(f"✅ Embedding generated: {len(query_embedding)} dimensions")
        print(f"   First 5 values: {query_embedding[:5]}")
        
        # Step 2: Load collection (exactly like your app)
        print("\n🧪 Step 2: Loading collection...")
        
        describe_url = f"{milvus_uri}/v2/vectordb/collections/describe"
        headers = {
            "Authorization": f"Bearer {milvus_token}",
            "Content-Type": "application/json"
        }
        
        describe_data = {"collectionName": "rss_feeds"}
        response = requests.post(describe_url, json=describe_data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            load_state = result.get('data', {}).get('load', 'Unknown')
            print(f"Collection load state: {load_state}")
            
            if load_state == "LoadStateNotLoad":
                print("Loading collection...")
                load_url = f"{milvus_uri}/v2/vectordb/collections/load"
                load_data = {"collectionName": "rss_feeds"}
                
                load_response = requests.post(load_url, json=load_data, headers=headers, timeout=10)
                if load_response.status_code == 200:
                    print("✅ Collection loaded successfully")
                else:
                    print(f"⚠️ Collection load failed: {load_response.status_code}")
            else:
                print("✅ Collection already loaded")
        else:
            print(f"❌ Failed to check collection status: {response.status_code}")
        
        # Step 3: Perform search (exactly like your app)
        print("\n🧪 Step 3: Performing vector search...")
        
        search_url = f"{milvus_uri}/v2/vectordb/entities/search"
        
        # Use the exact same format as your app
        output_fields = [
            "text_content", "article_title", "published_date", "feed_name", 
            "chunk_type", "companies", "products", "regulations", "regulatory_bodies"
        ]
        
        # Convert to float32 array exactly like your app
        query_embedding_float32 = np.array(query_embedding, dtype=np.float32).flatten().tolist()
        
        search_data = {
            "collectionName": "rss_feeds",
            "data": [query_embedding_float32],  # Fixed: wrapped in list
            "limit": 5,
            "outputFields": output_fields,
            "metricType": "COSINE",
            "params": {"nprobe": 10},
            "fieldName": "text_vector"
        }
        
        print(f"Search URL: {search_url}")
        print(f"Search data structure: {json.dumps(search_data, indent=2)}")
        
        search_response = requests.post(search_url, json=search_data, headers=headers, timeout=10)
        print(f"Search response status: {search_response.status_code}")
        
        if search_response.status_code == 200:
            result = search_response.json()
            print("✅ Search request successful")
            
            # Check for error codes
            if 'code' in result and result.get('code') != 0:
                print(f"❌ Milvus API error: Code {result.get('code')}, Message: {result.get('message')}")
                return False
            
            if 'data' in result and result['data']:
                print(f"🎉 Found {len(result['data'])} results!")
                
                # Show first result
                first_result = result['data'][0]
                print(f"\nFirst result:")
                print(f"  - Title: {first_result.get('article_title', 'N/A')}")
                print(f"  - Content length: {len(first_result.get('text_content', ''))}")
                print(f"  - Published: {first_result.get('published_date', 'N/A')}")
                
                return True
            else:
                print("❌ No data in search results")
                print(f"Full response: {json.dumps(result, indent=2)}")
                return False
        else:
            print(f"❌ Search failed: {search_response.status_code}")
            print(f"Response: {search_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during search process: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 App Search Process Test")
    print("=" * 50)
    
    success = asyncio.run(test_app_search_process())
    
    if success:
        print("\n🎉 All steps completed successfully!")
        print("   Your app should work now")
    else:
        print("\n❌ Search process failed")
        print("   Check the error details above")
    
    print("\n" + "=" * 50) 