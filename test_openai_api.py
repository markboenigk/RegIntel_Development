#!/usr/bin/env python3
"""
Simple script to test if OpenAI API key is still active
"""

import os
import openai
from dotenv import load_dotenv

def test_openai_api():
    """Test if OpenAI API key is active and working"""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Please check your .env file or set the environment variable")
        return False
    
    print(f"🔑 Found OpenAI API key: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple API call (using models endpoint to check authentication)
        print("🧪 Testing API key with OpenAI API...")
        
        # Get list of available models (this is a lightweight call)
        models = client.models.list()
        
        if models:
            print("✅ API key is ACTIVE and working!")
            print(f"   Available models: {len(models.data)} models found")
            
            # Show a few model examples
            model_names = [model.id for model in models.data[:5]]
            print(f"   Sample models: {', '.join(model_names)}")
            
            return True
        else:
            print("⚠️ API key appears valid but no models returned")
            return False
            
    except openai.AuthenticationError:
        print("❌ API key is INVALID or expired")
        print("   Please check your OpenAI account and regenerate the key if needed")
        return False
        
    except openai.RateLimitError:
        print("⚠️ Rate limit exceeded - API key is valid but you've hit usage limits")
        return False
        
    except openai.APIError as e:
        print(f"⚠️ OpenAI API error: {e}")
        print("   This might indicate an issue with the API service")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_simple_completion():
    """Test with a simple text completion to verify full functionality"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        print("\n🧪 Testing with a simple completion...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello, API test successful!' and nothing else."}
            ],
            max_tokens=20
        )
        
        if response.choices and response.choices[0].message.content:
            print("✅ Text completion test successful!")
            print(f"   Response: {response.choices[0].message.content.strip()}")
            return True
        else:
            print("⚠️ Completion test failed - no response content")
            return False
            
    except Exception as e:
        print(f"❌ Completion test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing OpenAI API Key Status")
    print("=" * 40)
    
    # Test basic API connectivity
    basic_test = test_openai_api()
    
    if basic_test:
        # If basic test passes, try a simple completion
        completion_test = test_simple_completion()
        
        if completion_test:
            print("\n🎉 All tests passed! Your OpenAI API key is fully functional.")
        else:
            print("\n⚠️ Basic API connectivity works, but completion test failed.")
    else:
        print("\n❌ API key test failed. Please check your configuration.")
    
    print("\n" + "=" * 40) 