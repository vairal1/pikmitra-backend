#!/usr/bin/env python3
"""
Simple test script to test the prediction API with a real image
"""

import requests
import base64
import json
from PIL import Image
import numpy as np
import io

def create_test_plant_image():
    """Create a simple test plant image (green rectangle representing a leaf)"""
    # Create a simple green image simulating a plant leaf
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    # Make it green-ish
    img_array[:, :, 1] = 100  # Green channel
    img_array[50:174, 50:174, 1] = 150  # Brighter green in the center
    
    # Add some brown spots to simulate disease
    img_array[80:100, 80:100, :] = [139, 69, 19]  # Brown spots
    img_array[120:140, 120:140, :] = [139, 69, 19]  # Brown spots
    
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def test_prediction():
    """Test the prediction endpoint"""
    print("🔬 Testing ML model prediction...")
    
    # Create test image
    test_image = create_test_plant_image()
    
    # Test prediction
    payload = {
        "image": test_image,
        "language": "en"
    }
    
    try:
        print("Making prediction request...")
        response = requests.post("http://localhost:5000/predict-base64", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ REAL Prediction successful!")
                print(f"   🌿 Disease: {data.get('disease')}")
                print(f"   📊 Confidence: {data.get('confidence', 0):.3f}")
                print(f"   🔑 Disease Key: {data.get('disease_key')}")
                print(f"   💊 Cure: {data.get('cure', 'No cure info')[:100]}...")
                print(f"   🗣️ Language: {data.get('language')}")
                return True
            else:
                print(f"❌ Prediction failed: {data.get('message')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("\n🎉 ML model is working correctly and making REAL predictions!")
    else:
        print("\n⚠️ ML model test failed - check server logs for details")