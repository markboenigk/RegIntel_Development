#!/usr/bin/env python3
"""
Simple local server for development without uvicorn
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from index import app
    import uvicorn
    
    if __name__ == "__main__":
        print("🚀 Starting local development server...")
        print("📱 Open http://localhost:8000 in your browser")
        print("🔐 Test auth at http://localhost:8000/api/auth/status")
        print("📝 Press Ctrl+C to stop the server")
        
        uvicorn.run(
            "index:app",
            host="127.0.0.1",  # Use 127.0.0.1 instead of 0.0.0.0
            port=8000,
            reload=True,
            log_level="info"
        )
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Try installing requirements: pip3 install -r requirements.txt")
    print("💡 Or use the simple server below:")
    
    # Fallback: Simple HTTP server for static files
    print("\n🔄 Starting simple HTTP server for static files...")
    os.system("python3 -m http.server 8000")
    print("📱 Open http://localhost:8000 in your browser") 