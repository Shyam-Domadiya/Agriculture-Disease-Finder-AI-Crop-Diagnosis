"""
Local runner for Agriculture Disease Finder API.
Imports the FastAPI app instance from api/index.py (which runs as a Vercel serverless function).
"""

import uvicorn
import sys
import os

# Add root directory to python path to ensure clean imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.index import app

if __name__ == "__main__":
    print("🌿 Starting local dev server...")
    print("📡 Local API Docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
