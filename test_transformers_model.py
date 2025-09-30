#!/usr/bin/env python3
"""
Test the transformers model directly with different inputs
"""

from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
import numpy as np
from PIL import Image
import requests

def create_test_images():
    """Create different test images"""
    images = []
    
    # Image 1: Green leaf
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 1] = 150  # Green
    img_array[50:174, 50:174, 1] = 200
    images.append(Image.fromarray(img_array))
    
    # Image 2: Red tomato-like
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 0] = 200  # Red
    img_array[60:164, 60:164, 0] = 250
    images.append(Image.fromarray(img_array))
    
    # Image 3: Yellow/brown diseased
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 0] = 180  # Mix for yellow/brown
    img_array[:, :, 1] = 150
    img_array[:, :, 2] = 50
    # Brown spots
    img_array[80:120, 80:120] = [100, 50, 20]
    images.append(Image.fromarray(img_array))
    
    # Image 4: Pure random noise
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    images.append(Image.fromarray(img_array))
    
    # Image 5: Checkerboard pattern
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 32):
        for j in range(0, 224, 32):
            if (i//32 + j//32) % 2:
                img_array[i:i+32, j:j+32] = [255, 255, 255]
            else:
                img_array[i:i+32, j:j+32] = [0, 100, 0]
    images.append(Image.fromarray(img_array))
    
    return images

def test_transformers_model():
    print("🔬 Testing Transformers Model Directly")
    print("=" * 50)
    
    # Load model and processor
    model_path = '../model/mobilenet_v2_1.0_224-plant-disease-identification'
    print(f"Loading model from: {model_path}")
    
    model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Number of classes: {model.config.num_labels}")
    print(f"🔧 Model in eval mode: {not model.training}")
    
    # Create test images
    test_images = create_test_images()
    image_descriptions = [
        "Green Leaf Pattern",
        "Red Tomato Pattern", 
        "Yellow/Brown Diseased",
        "Random Noise",
        "Checkerboard Pattern"
    ]
    
    results = []
    
    for i, (image, description) in enumerate(zip(test_images, image_descriptions)):
        print(f"\n--- Test {i+1}: {description} ---")
        
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")
        print(f"Input shape: {inputs['pixel_values'].shape}")
        print(f"Input stats: mean={inputs['pixel_values'].mean():.4f}, std={inputs['pixel_values'].std():.4f}")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]  # Remove batch dimension
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
            
            confidence_val = confidence.item()
            predicted_class_val = int(predicted_class.item())
            
            # Get label
            label = model.config.id2label.get(predicted_class_val, f"Unknown_Class_{predicted_class_val}")
            
            print(f"Predicted class: {predicted_class_val}")
            print(f"Label: {label}")
            print(f"Confidence: {confidence_val:.6f}")
            
            results.append({
                'image': description,
                'class': predicted_class_val,
                'label': label,
                'confidence': confidence_val
            })
            
            # Show top 3 for first image
            if i == 0:
                top3_conf, top3_classes = torch.topk(probabilities, 3)
                print("Top 3 predictions:")
                for j in range(3):
                    class_idx = top3_classes[j].item()
                    conf = top3_conf[j].item()
                    class_label = model.config.id2label.get(class_idx, f"Unknown_Class_{class_idx}")
                    print(f"  {j+1}. Class {class_idx} ({class_label}): {conf:.6f}")
    
    # Analysis
    print("\n" + "=" * 50)
    print("📊 ANALYSIS")
    print("=" * 50)
    
    unique_classes = set(r['class'] for r in results)
    unique_labels = set(r['label'] for r in results)
    unique_confidences = set(round(r['confidence'], 4) for r in results)
    
    print(f"Unique classes: {len(unique_classes)} - {sorted(unique_classes)}")
    print(f"Unique labels: {len(unique_labels)} - {list(unique_labels)}")
    print(f"Unique confidences: {len(unique_confidences)}")
    
    confidence_range = (min(r['confidence'] for r in results), max(r['confidence'] for r in results))
    print(f"Confidence range: {confidence_range[0]:.6f} - {confidence_range[1]:.6f}")
    
    # Summary table
    print(f"\n📋 RESULTS:")
    for r in results:
        print(f"{r['image']:<25} Class {r['class']:<2} {r['label']:<35} {r['confidence']:.4f}")
    
    # Verdict
    if len(unique_classes) >= 3 and len(unique_confidences) >= 3:
        print("\n✅ SUCCESS: Model produces diverse predictions!")
    elif len(unique_classes) >= 2:
        print("\n⚠️ PARTIAL: Some diversity in predictions")
    else:
        print("\n❌ PROBLEM: Model predicts same class for all inputs")
        print("   This suggests the model might be broken or poorly trained")

if __name__ == "__main__":
    test_transformers_model()