#!/usr/bin/env python3
"""
Test script to verify Milvus connection and identify connection issues
"""

import os
import requests
from dotenv import load_dotenv

def test_milvus_connection():
    """Test Milvus connection and identify issues"""
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_token = os.getenv("MILVUS_TOKEN")
    
    print("🔍 Testing Milvus Connection")
    print("=" * 40)
    print(f"URI: {milvus_uri}")
    print(f"Token: {milvus_token[:20]}..." if milvus_token else "None")
    
    if not milvus_uri or not milvus_token:
        print("❌ Missing Milvus credentials")
        return False
    
    try:
        # Test 1: Check if we can reach the Milvus endpoint
        print("\n🧪 Test 1: Checking endpoint reachability...")
        
        # Test basic connectivity
        test_url = f"{milvus_uri}/v2/vectordb/collections/describe"
        headers = {
            "Authorization": f"Bearer {milvus_token}",
            "Content-Type": "application/json"
        }
        
        print(f"Testing endpoint: {test_url}")
        
        response = requests.post(test_url, json={}, headers=headers, timeout=10)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Endpoint is reachable and responding")
            result = response.json()
            print(f"Response: {result}")
        else:
            print(f"❌ Endpoint returned error: {response.status_code}")
            print(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("   This usually means the URI is incorrect or the service is down")
        return False
        
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout error: {e}")
        print("   The service is taking too long to respond")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    try:
        # Test 2: Check if the rss_feeds collection exists
        print("\n🧪 Test 2: Checking rss_feeds collection...")
        
        describe_data = {"collectionName": "rss_feeds"}
        response = requests.post(test_url, json=describe_data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Collection exists")
            print(f"Collection info: {result}")
            
            # Check if collection is loaded
            if 'data' in result and 'load' in result['data']:
                load_state = result['data']['load']
                print(f"Load state: {load_state}")
                
                if load_state == "LoadStateNotLoad":
                    print("⚠️ Collection is not loaded - this might cause issues")
                else:
                    print("✅ Collection is loaded and ready")
            else:
                print("⚠️ Could not determine collection load state")
                
        else:
            print(f"❌ Failed to describe collection: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking collection: {e}")
        return False
    
    try:
        # Test 3: Try a simple search to see if the collection has data
        print("\n🧪 Test 3: Testing search functionality...")
        
        # First, try to load the collection if it's not loaded
        load_url = f"{milvus_uri}/v2/vectordb/collections/load"
        load_data = {"collectionName": "rss_feeds"}
        
        print("Attempting to load collection...")
        load_response = requests.post(load_url, json=load_data, headers=headers, timeout=10)
        
        if load_response.status_code == 200:
            load_result = load_response.json()
            if load_result.get('code') == 0:
                print("✅ Collection loaded successfully")
            else:
                print(f"⚠️ Collection load returned code: {load_result.get('code')}")
        else:
            print(f"⚠️ Failed to load collection: {load_response.status_code}")
        
        # Now try a simple search
        search_url = f"{milvus_uri}/v2/vectordb/entities/search"
        search_data = {
            "collectionName": "rss_feeds",
            "data": [[0.1] * 1536],  # Dummy embedding
            "limit": 1,
            "outputFields": ["text", "metadata"]
        }
        
        print("Testing search with dummy embedding...")
        search_response = requests.post(search_url, json=search_data, headers=headers, timeout=10)
        
        if search_response.status_code == 200:
            result = search_response.json()
            print("✅ Search endpoint is working")
            
            if 'data' in result and result['data']:
                print(f"Found {len(result['data'])} results")
                print("✅ Collection has data and search is functional")
            else:
                print("⚠️ Search returned no results - collection might be empty")
                
        else:
            print(f"❌ Search failed: {search_response.status_code}")
            print(f"Response: {search_response.text}")
            
    except Exception as e:
        print(f"❌ Error testing search: {e}")
        return False
    
    print("\n🎉 Milvus connection test completed!")
    return True

if __name__ == "__main__":
    test_milvus_connection() 