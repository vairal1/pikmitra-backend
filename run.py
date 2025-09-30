#!/usr/bin/env python3
"""
Simple script to run the Flask app
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    
    print("🚀 Starting Pikmitra Flask Backend...")
    print("🌐 Backend will be available at: http://localhost:5000")
    print("📊 Health check: http://localhost:5000/health")
    print("=" * 50)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
    
except Exception as e:
    print(f"❌ Error starting Flask app: {e}")
    sys.exit(1)
