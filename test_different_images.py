#!/usr/bin/env python3
"""
Test script to verify that different plant images produce different ML predictions
"""

import requests
import base64
import json
from PIL import Image
import numpy as np
import io
import time

def create_plant_image(image_type="healthy_green"):
    """Create different types of synthetic plant images"""
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    
    if image_type == "healthy_green":
        # Healthy green leaf
        img_array[:, :, 1] = 120  # Green channel
        img_array[50:174, 50:174, 1] = 180  # Brighter green center
        
    elif image_type == "brown_spots":
        # Leaf with brown disease spots
        img_array[:, :, 1] = 100  # Base green
        img_array[50:174, 50:174, 1] = 150  # Green center
        # Add brown spots (disease-like)
        img_array[60:90, 60:90, :] = [139, 69, 19]  # Brown spot 1
        img_array[130:160, 130:160, :] = [160, 82, 45]  # Brown spot 2
        img_array[80:100, 140:160, :] = [101, 67, 33]  # Brown spot 3
        
    elif image_type == "yellow_diseased":
        # Yellowing diseased leaf
        img_array[:, :, 0] = 200  # Red channel for yellow
        img_array[:, :, 1] = 200  # Green channel for yellow
        img_array[:, :, 2] = 50   # Low blue for yellow
        # Add some brown edges
        img_array[:20, :, :] = [139, 69, 19]   # Top edge
        img_array[-20:, :, :] = [139, 69, 19]  # Bottom edge
        
    elif image_type == "red_fruit":
        # Reddish fruit/tomato-like
        img_array[:, :, 0] = 180  # High red
        img_array[:, :, 1] = 50   # Low green
        img_array[:, :, 2] = 50   # Low blue
        # Add some darker spots
        img_array[80:120, 80:120, :] = [100, 30, 30]  # Dark red spot
        
    elif image_type == "mixed_colors":
        # Mixed colored leaf (complex pattern)
        # Green base
        img_array[:, :, 1] = 120
        # Brown sections
        img_array[0:74, 0:112, :] = [139, 69, 19]
        # Yellow sections
        img_array[150:224, 0:112, 0] = 200
        img_array[150:224, 0:112, 1] = 200
        img_array[150:224, 0:112, 2] = 50
        
    # Add some texture/noise for realism
    noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def test_prediction(image_data, image_name):
    """Test prediction for a single image"""
    payload = {
        "image": image_data,
        "language": "en"
    }
    
    try:
        print(f"\n🔬 Testing {image_name}...")
        response = requests.post("http://localhost:5000/predict-base64", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return {
                    'name': image_name,
                    'crop': data.get('crop', 'Unknown'),
                    'disease': data.get('disease', 'Unknown'),
                    'confidence': data.get('confidence', 0),
                    'disease_key': data.get('disease_key', 'N/A'),
                    'cure_available': bool(data.get('cure') and data.get('cure') != 'No cure information available for this disease.')
                }
            else:
                print(f"❌ API error: {data.get('message')}")
                return None
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the Flask server running?")
        return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    """Run comprehensive test with different plant images"""
    print("🌱 Testing Plant Disease Detection with Different Images")
    print("=" * 60)
    
    # Create different types of images
    test_images = [
        ("healthy_green", "Healthy Green Leaf"),
        ("brown_spots", "Leaf with Brown Disease Spots"),
        ("yellow_diseased", "Yellow Diseased Leaf"),
        ("red_fruit", "Red Fruit/Tomato"),
        ("mixed_colors", "Mixed Color Diseased Leaf")
    ]
    
    results = []
    
    for image_type, description in test_images:
        print(f"\n📸 Generating {description}...")
        image_data = create_plant_image(image_type)
        
        # Test prediction
        result = test_prediction(image_data, description)
        if result:
            results.append(result)
            print(f"✅ Success!")
            print(f"   🌿 Crop: {result['crop']}")
            print(f"   🦠 Disease: {result['disease']}")
            print(f"   📊 Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
            print(f"   🔑 Disease Key: {result['disease_key']}")
            print(f"   💊 Cure Available: {'Yes' if result['cure_available'] else 'No'}")
        else:
            print(f"❌ Failed!")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📊 ANALYSIS RESULTS")
    print("=" * 60)
    
    if not results:
        print("❌ No successful predictions - check server connection!")
        return
    
    print(f"✅ Total successful predictions: {len(results)}")
    
    # Check for diversity
    unique_crops = set(r['crop'] for r in results)
    unique_diseases = set(r['disease'] for r in results)
    unique_keys = set(r['disease_key'] for r in results)
    
    print(f"🌿 Unique crops detected: {len(unique_crops)} - {list(unique_crops)}")
    print(f"🦠 Unique diseases detected: {len(unique_diseases)} - {list(unique_diseases)}")
    print(f"🔑 Unique disease keys: {len(unique_keys)}")
    
    # Check confidence ranges
    confidences = [r['confidence'] for r in results]
    print(f"📊 Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
    print(f"📊 Average confidence: {np.mean(confidences):.3f}")
    
    # Summary table
    print(f"\n📋 DETAILED RESULTS:")
    print(f"{'Image Type':<25} {'Crop':<12} {'Disease':<20} {'Conf%':<8} {'Cure'}")
    print("-" * 75)
    for r in results:
        cure_status = "✅" if r['cure_available'] else "❌"
        print(f"{r['name']:<25} {r['crop']:<12} {r['disease']:<20} {r['confidence']*100:<7.1f}% {cure_status}")
    
    # Final verdict
    print(f"\n🎯 VERDICT:")
    if len(unique_diseases) >= 3:
        print("✅ SUCCESS: Different images produce DIFFERENT predictions!")
        print("   ML model is working correctly and not returning static responses.")
    elif len(unique_diseases) >= 2:
        print("⚠️  PARTIAL: Some diversity in predictions, but could be better.")
        print("   ML model is working but might need more diverse test images.")
    else:
        print("❌ FAILURE: All images produce SAME predictions!")
        print("   This suggests the model might be returning cached/static responses.")
    
    print(f"\n🔄 Note: With synthetic test images, some similarity in predictions is normal.")
    print(f"🔄 For best results, test with real plant photos showing different diseases.")

if __name__ == "__main__":
    main()