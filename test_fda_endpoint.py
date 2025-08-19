#!/usr/bin/env python3
"""
Test script to test the FDA warning letters endpoint
"""

import requests
import json

def test_fda_endpoint():
    """Test the FDA warning letters chat endpoint"""
    
    url = "http://localhost:8000/api/chat/fda_warning_letters"
    
    payload = {
        "message": "What is the warning letter from Artisan Vapor Company about?",
        "conversation_history": []
    }
    
    print("🔍 Testing FDA Warning Letters API Endpoint")
    print("=" * 50)
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"\n📡 Response Status: {response.status_code}")
        print(f"📡 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Response: {json.dumps(data, indent=2)}")
            
            # Check if we got sources
            if data.get('sources'):
                print(f"\n📚 Sources found: {len(data['sources'])}")
                for i, source in enumerate(data['sources'][:3], 1):
                    print(f"  Source {i}: {source.get('title', 'No title')}")
            else:
                print("❌ No sources returned")
                
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
    test_fda_endpoint() 