#!/usr/bin/env python3
"""
Test with properly preprocessed plant-like images to see if preprocessing is the issue
"""

import requests
import base64
import json
from PIL import Image
import numpy as np
import io
import time

def create_realistic_plant_image(image_type="healthy_leaf"):
    """Create more realistic plant images with proper preprocessing"""
    
    if image_type == "healthy_leaf":
        # Create a green leaf-like image
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        # Green gradient
        for y in range(224):
            for x in range(224):
                # Distance from center for leaf shape
                dist = ((x - 112)**2 + (y - 112)**2) ** 0.5
                if dist < 100:  # Leaf shape
                    green_intensity = max(50, 200 - int(dist * 1.5))
                    img_array[y, x] = [20, green_intensity, 30]
                else:
                    img_array[y, x] = [240, 240, 240]  # White background
    
    elif image_type == "diseased_leaf":
        # Create a diseased leaf with brown spots
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        # Base green
        for y in range(224):
            for x in range(224):
                dist = ((x - 112)**2 + (y - 112)**2) ** 0.5
                if dist < 100:
                    img_array[y, x] = [30, 120, 40]
                else:
                    img_array[y, x] = [240, 240, 240]
        
        # Add brown disease spots
        spots = [(60, 60, 15), (150, 80, 20), (100, 140, 18), (130, 120, 12)]
        for sx, sy, radius in spots:
            for y in range(max(0, sy-radius), min(224, sy+radius)):
                for x in range(max(0, sx-radius), min(224, sx+radius)):
                    dist = ((x - sx)**2 + (y - sy)**2) ** 0.5
                    if dist < radius:
                        img_array[y, x] = [139, 69, 19]  # Brown
    
    elif image_type == "red_tomato":
        # Create a red tomato-like image
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        # Red circular shape
        for y in range(224):
            for x in range(224):
                dist = ((x - 112)**2 + (y - 112)**2) ** 0.5
                if dist < 80:
                    red_intensity = max(150, 250 - int(dist * 1.2))
                    img_array[y, x] = [red_intensity, 50, 30]
                else:
                    img_array[y, x] = [240, 240, 240]
                    
        # Add some spots/texture
        for i in range(5):
            spot_x = np.random.randint(80, 144)
            spot_y = np.random.randint(80, 144)
            img_array[spot_y-3:spot_y+3, spot_x-3:spot_x+3] = [100, 30, 20]
    
    elif image_type == "yellow_corn":
        # Create a yellow corn-like pattern
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        # Yellow base
        img_array[:, :] = [240, 240, 240]  # White background
        # Add corn kernel pattern
        for y in range(50, 174, 15):
            for x in range(50, 174, 15):
                # Draw kernel-like oval
                for dy in range(-6, 7):
                    for dx in range(-4, 5):
                        if dy*dy/36 + dx*dx/16 <= 1:
                            if y+dy < 224 and x+dx < 224:
                                img_array[y+dy, x+dx] = [200, 180, 50]
    
    elif image_type == "apple_fruit":
        # Create a red/green apple
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        # Background
        img_array[:, :] = [240, 240, 240]
        # Apple shape (circular-ish)
        for y in range(224):
            for x in range(224):
                dist = ((x - 112)**2 + (y - 100)**2) ** 0.5
                if dist < 70:
                    # Mix of red and green
                    if x < 112:
                        img_array[y, x] = [180, 50, 40]  # Red side
                    else:
                        img_array[y, x] = [80, 150, 60]  # Green side
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def test_realistic_images():
    """Test with more realistic plant-like images"""
    print("🌱 Testing with Realistic Plant Images")
    print("=" * 60)
    
    test_images = [
        ("healthy_leaf", "Healthy Green Leaf"),
        ("diseased_leaf", "Diseased Leaf with Brown Spots"),
        ("red_tomato", "Red Tomato Fruit"),
        ("yellow_corn", "Yellow Corn Pattern"),
        ("apple_fruit", "Red-Green Apple")
    ]
    
    results = []
    
    for image_type, description in test_images:
        print(f"\n📸 Testing {description}...")
        
        # Create realistic image
        image_data = create_realistic_plant_image(image_type)
        
        # Make prediction
        payload = {
            "image": image_data,
            "language": "en"
        }
        
        try:
            response = requests.post("http://localhost:5000/predict-base64", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = {
                        'name': description,
                        'crop': data.get('crop', 'Unknown'),
                        'disease': data.get('disease', 'Unknown'),
                        'confidence': data.get('confidence', 0),
                        'full_name': data.get('disease_full', 'N/A')
                    }
                    results.append(result)
                    
                    print(f"✅ Success!")
                    print(f"   🌿 Crop: {result['crop']}")
                    print(f"   🦠 Disease: {result['disease']}")
                    print(f"   📊 Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
                    print(f"   📝 Full: {result['full_name']}")
                else:
                    print(f"❌ API Error: {data.get('message')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection failed. Is the Flask server running?")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(0.5)  # Small delay
    
    # Analysis
    if results:
        print("\n" + "=" * 60)
        print("📊 ANALYSIS")
        print("=" * 60)
        
        unique_crops = set(r['crop'] for r in results)
        unique_diseases = set(r['disease'] for r in results)
        unique_confidences = set(round(r['confidence'], 4) for r in results)
        
        print(f"✅ Total predictions: {len(results)}")
        print(f"🌿 Unique crops: {len(unique_crops)} - {list(unique_crops)}")
        print(f"🦠 Unique diseases: {len(unique_diseases)} - {list(unique_diseases)}")
        print(f"📊 Unique confidences: {len(unique_confidences)}")
        
        confidence_range = (min(r['confidence'] for r in results), max(r['confidence'] for r in results))
        print(f"📊 Confidence range: {confidence_range[0]:.4f} - {confidence_range[1]:.4f}")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print(f"{'Image':<30} {'Crop':<15} {'Disease':<20} {'Conf%':<8}")
        print("-" * 80)
        for r in results:
            print(f"{r['name']:<30} {r['crop']:<15} {r['disease']:<20} {r['confidence']*100:<7.2f}%")
        
        # Verdict
        print(f"\n🎯 VERDICT:")
        if len(unique_diseases) >= 3 and len(unique_confidences) >= 3:
            print("✅ EXCELLENT: Model produces diverse predictions with varying confidence!")
        elif len(unique_diseases) >= 2:
            print("⚠️ GOOD: Some diversity in predictions")
        else:
            print("❌ PROBLEM: Model still produces similar predictions")
            print("   This suggests an issue with the model weights or architecture")

if __name__ == "__main__":
    test_realistic_images()