#!/usr/bin/env python3
"""
Test the backend API with synthetic images
"""

import requests
import base64
import numpy as np
from PIL import Image
import io

def create_test_image(image_type):
    """Create different test images"""
    
    if image_type == "green_leaf":
        # Green leaf
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 1] = 150  # Green
        img_array[50:174, 50:174, 1] = 200
    elif image_type == "red_tomato":
        # Red tomato-like
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 0] = 200  # Red
        img_array[60:164, 60:164, 0] = 250
    elif image_type == "yellow_diseased":
        # Yellow/brown diseased
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 0] = 180  # Mix for yellow/brown
        img_array[:, :, 1] = 150
        img_array[:, :, 2] = 50
        # Brown spots
        img_array[80:120, 80:120] = [100, 50, 20]
    elif image_type == "random_noise":
        # Pure random noise
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    else:
        # Default: checkerboard
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(0, 224, 32):
            for j in range(0, 224, 32):
                if (i//32 + j//32) % 2:
                    img_array[i:i+32, j:j+32] = [255, 255, 255]
                else:
                    img_array[i:i+32, j:j+32] = [0, 100, 0]
    
    return Image.fromarray(img_array)

def image_to_base64(image):
    """Convert PIL image to base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_backend_api():
    print("🧪 Testing Backend API with Synthetic Images")
    print("=" * 50)
    
    # Backend URL
    api_url = "http://localhost:5000/predict-base64"
    
    # Test image types
    test_cases = [
        ("green_leaf", "Green Leaf Pattern"),
        ("red_tomato", "Red Tomato Pattern"),
        ("yellow_diseased", "Yellow/Brown Diseased"),
        ("random_noise", "Random Noise"),
        ("checkerboard", "Checkerboard Pattern")
    ]
    
    results = []
    
    for image_type, description in test_cases:
        print(f"\n--- Testing: {description} ---")
        
        # Create test image
        image = create_test_image(image_type)
        image_base64 = image_to_base64(image)
        
        # Prepare request
        payload = {
            "image": image_base64
        }
        
        try:
            # Make API request
            response = requests.post(api_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success:")
                print(f"  Crop: {result.get('crop', 'N/A')}")
                print(f"  Disease: {result.get('disease', 'N/A')}")
                print(f"  Confidence: {result.get('confidence', 'N/A')}")
                print(f"  Cure: {result.get('cure', 'N/A')[:50]}..." if result.get('cure') else "  Cure: N/A")
                
                results.append({
                    'image_type': description,
                    'crop': result.get('crop', 'N/A'),
                    'disease': result.get('disease', 'N/A'), 
                    'confidence': result.get('confidence', 0)
                })
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            print("Make sure the backend server is running on localhost:5000")
    
    # Analysis
    print("\n" + "=" * 50)
    print("📊 ANALYSIS")
    print("=" * 50)
    
    if results:
        unique_crops = set(r['crop'] for r in results)
        unique_diseases = set(r['disease'] for r in results)
        unique_confidences = set(round(r['confidence'], 2) for r in results)
        
        print(f"Unique crops: {len(unique_crops)} - {sorted(unique_crops)}")
        print(f"Unique diseases: {len(unique_diseases)} - {sorted(unique_diseases)}")
        print(f"Unique confidences: {len(unique_confidences)}")
        
        print(f"\n📋 RESULTS:")
        for r in results:
            print(f"{r['image_type']:<25} {r['crop']:<15} {r['disease']:<30} {r['confidence']:.2f}")
        
        # Verdict
        if len(unique_crops) >= 3 and len(unique_diseases) >= 3:
            print("\n✅ SUCCESS: API produces diverse predictions!")
        elif len(unique_crops) >= 2 or len(unique_diseases) >= 2:
            print("\n⚠️ PARTIAL: Some diversity in predictions")  
        else:
            print("\n❌ PROBLEM: API predicts same result for all inputs")
    else:
        print("No successful results to analyze")

if __name__ == "__main__":
    test_backend_api()