#!/usr/bin/env python3
"""
Test script to directly call the chat API endpoint and identify the error
"""

import requests
import json

def test_chat_endpoint():
    """Test the chat endpoint directly"""
    
    url = "http://localhost:8000/api/chat/rss_feeds"
    
    payload = {
        "message": "What news about Stryker?",
        "conversation_history": []
    }
    
    print("🔍 Testing Chat API Endpoint")
    print("=" * 40)
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"\n📡 Response Status: {response.status_code}")
        print(f"📡 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Error Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        print(f"Raw response: {response.text}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_chat_endpoint() 