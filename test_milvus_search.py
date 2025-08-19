#!/usr/bin/env python3
"""
Test script to debug Milvus search and see why no results are returned
"""

import os
import requests
import json
from dotenv import load_dotenv

def test_milvus_search():
    """Test Milvus search with detailed debugging"""
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_token = os.getenv("MILVUS_TOKEN")
    
    print("🔍 Testing Milvus Search with Detailed Debugging")
    print("=" * 50)
    
    if not milvus_uri or not milvus_token:
        print("❌ Missing Milvus credentials")
        return False
    
    try:
        # Test 1: Check collection statistics
        print("\n🧪 Test 1: Checking collection statistics...")
        
        stats_url = f"{milvus_uri}/v2/vectordb/collections/describe"
        headers = {
            "Authorization": f"Bearer {milvus_token}",
            "Content-Type": "application/json"
        }
        
        # Check rss_feeds collection
        describe_data = {"collectionName": "rss_feeds"}
        response = requests.post(stats_url, json=describe_data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Collection description retrieved")
            
            # Look for entity count
            if 'data' in result:
                print(f"Collection data: {json.dumps(result['data'], indent=2)}")
                
                # Check if there are any entities
                if 'entityCount' in result['data']:
                    entity_count = result['data']['entityCount']
                    print(f"📊 Entity count: {entity_count}")
                    
                    if entity_count == 0:
                        print("❌ Collection is empty - no entities found")
                        return False
                    else:
                        print(f"✅ Collection has {entity_count} entities")
                else:
                    print("⚠️ Could not determine entity count")
            else:
                print("⚠️ No data field in response")
        else:
            print(f"❌ Failed to get collection description: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Test 2: Try a simple search with minimal parameters
        print("\n🧪 Test 2: Testing simple search...")
        
        search_url = f"{milvus_uri}/v2/vectordb/entities/search"
        
        # Create a simple test embedding (1536 dimensions)
        test_embedding = [0.1] * 1536
        
        search_data = {
            "collectionName": "rss_feeds",
            "data": [test_embedding],  # Note: embedding should be in a list
            "limit": 5,
            "outputFields": ["text_content", "article_title", "published_date"],
            "metricType": "COSINE"
        }
        
        print(f"Search URL: {search_url}")
        print(f"Search data: {json.dumps(search_data, indent=2)}")
        
        search_response = requests.post(search_url, json=search_data, headers=headers, timeout=10)
        print(f"Search response status: {search_response.status_code}")
        
        if search_response.status_code == 200:
            result = search_response.json()
            print("✅ Search request successful")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if 'data' in result and result['data']:
                print(f"🎉 Found {len(result['data'])} results!")
                
                # Show first result details
                first_result = result['data'][0]
                print(f"\nFirst result:")
                print(f"  - text_content length: {len(first_result.get('text_content', ''))}")
                print(f"  - article_title: {first_result.get('article_title', 'N/A')}")
                print(f"  - published_date: {first_result.get('published_date', 'N/A')}")
                
                return True
            else:
                print("❌ No data in search results")
                print("This suggests the search is working but not finding matches")
                return False
        else:
            print(f"❌ Search failed: {search_response.status_code}")
            print(f"Response: {search_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during search test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_collection_data():
    """Test if we can get any data from the collection"""
    
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_token = os.getenv("MILVUS_TOKEN")
    
    print("\n🧪 Test 3: Testing collection data retrieval...")
    
    try:
        # Try to get some entities directly
        query_url = f"{milvus_uri}/v2/vectordb/entities/query"
        headers = {
            "Authorization": f"Bearer {milvus_token}",
            "Content-Type": "application/json"
        }
        
        query_data = {
            "collectionName": "rss_feeds",
            "outputFields": ["text_content", "article_title", "published_date"],
            "limit": 3
        }
        
        print(f"Query URL: {query_url}")
        print(f"Query data: {json.dumps(query_data, indent=2)}")
        
        response = requests.post(query_url, json=query_data, headers=headers, timeout=10)
        print(f"Query response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query successful")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if 'data' in result and result['data']:
                print(f"🎉 Found {len(result['data'])} entities in collection!")
                return True
            else:
                print("❌ No entities found in collection")
                return False
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during query test: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Milvus Search Debug Test")
    print("=" * 50)
    
    # Test search functionality
    search_test = test_milvus_search()
    
    if not search_test:
        # If search fails, try direct query
        data_test = test_collection_data()
        
        if data_test:
            print("\n✅ Collection has data but search is failing")
            print("   This suggests a search configuration issue")
        else:
            print("\n❌ Collection appears to be empty")
            print("   No entities found in the collection")
    
    print("\n" + "=" * 50) 