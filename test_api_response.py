#!/usr/bin/env python3
"""
Test script to show the exact API response format for plant disease detection
"""

import requests
import base64
import json
from PIL import Image
import numpy as np
import io

def create_test_plant_image():
    """Create a simple test plant image"""
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 1] = 100  # Green channel
    img_array[50:174, 50:174, 1] = 150  # Brighter green in center
    img_array[80:100, 80:100, :] = [139, 69, 19]  # Brown spots (disease)
    
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_api_response():
    """Test and display the exact API response"""
    print("🔬 Testing Plant Disease Detection API...")
    
    payload = {
        "image": create_test_plant_image(),
        "language": "en"
    }
    
    try:
        response = requests.post("http://localhost:5000/predict-base64", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            print("📋 EXACT API RESPONSE:")
            print("=" * 50)
            print(json.dumps(data, indent=2))
            print("=" * 50)
            
            if data.get('success'):
                print("\n✅ SUCCESS! Real ML predictions are working!")
                print(f"🌿 Crop: {data.get('disease', '').split(' with ')[0] if ' with ' in data.get('disease', '') else data.get('disease', '').split(' ')[0]}")
                print(f"🦠 Disease: {data.get('disease', '').split(' with ')[1] if ' with ' in data.get('disease', '') else 'Healthy' if 'healthy' in data.get('disease', '').lower() else data.get('disease', '')}")
                print(f"📊 Confidence: {data.get('confidence', 0):.3f}")
                print(f"🔑 Disease Key: {data.get('disease_key', 'N/A')}")
                print(f"💊 Has Cure Info: {'Yes' if data.get('cure') and data.get('cure') != 'No cure information available for this disease.' else 'No'}")
                
                return True
            else:
                print(f"❌ API returned error: {data.get('message')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the Flask server running on localhost:5000?")
        print("💡 Start it with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    test_api_response()