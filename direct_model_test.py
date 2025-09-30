#!/usr/bin/env python3
"""
Direct test of model inference with different random inputs
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import json
import numpy as np

# Add current directory to path
sys.path.append('.')

def test_model_directly():
    """Test the model directly without going through the app layer"""
    print("🔬 Direct Model Inference Test")
    print("=" * 50)
    
    # Define paths
    backend_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(backend_dir)
    model_path = os.path.join(project_root, 'model', 'mobilenet_v2_1.0_224-plant-disease-identification')
    
    print(f"Model path: {model_path}")
    
    # Load config first to get number of classes
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
        print("❌ Config file not found!")
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        id2label = config.get('id2label', {})
        num_classes = len(id2label)
        
    print(f"Number of classes: {num_classes}")
    print(f"Sample labels: {list(id2label.items())[:5]}")
    
    # Load model weights
    weights_path = os.path.join(model_path, 'pytorch_model.bin')
    if not os.path.exists(weights_path):
        print("❌ Model weights file not found!")
        return
        
    print("Loading model weights...")
    loaded_obj = torch.load(weights_path, map_location='cpu')
    
    # Handle different weight formats
    if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj:
        state_dict = loaded_obj['state_dict']
    elif isinstance(loaded_obj, dict):
        state_dict = loaded_obj
    else:
        # Assume it's a direct model
        model = loaded_obj
        model.eval()
        state_dict = None
    
    if state_dict is not None:
        # Create model and load weights
        backbone = mobilenet_v2(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
        print(f"Model classifier shape: {backbone.classifier[1]}")
        
        # Load state dict
        try:
            backbone.load_state_dict(state_dict, strict=False)
            print("✅ State dict loaded successfully")
        except Exception as e:
            print(f"⚠️ State dict loading issue: {e}")
            # Try loading without strict mode
            missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
        
        backbone.eval()
        model = backbone
    
    print("✅ Model loaded successfully")
    
    # Test with different random inputs
    print("\n🧪 Testing with different random inputs...")
    
    results = []
    for i in range(10):
        # Create completely different random tensors
        if i < 5:
            # Different random distributions
            img_tensor = torch.randn(1, 3, 224, 224) * (0.5 + i * 0.1)
        else:
            # Different patterns
            img_tensor = torch.ones(1, 3, 224, 224) * (0.1 * i) + torch.randn(1, 3, 224, 224) * 0.1
        
        print(f"\n--- Test {i+1} ---")
        print(f"Input stats: mean={img_tensor.mean():.4f}, std={img_tensor.std():.4f}, min={img_tensor.min():.4f}, max={img_tensor.max():.4f}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
            
            confidence_val = confidence.item()
            predicted_class_val = int(predicted_class.item())
            
            # Get disease label
            disease_label = id2label.get(str(predicted_class_val), f"Class_{predicted_class_val}")
            
            print(f"Predicted class: {predicted_class_val} ({disease_label})")
            print(f"Confidence: {confidence_val:.6f}")
            
            # Store result
            results.append({
                'test': i+1,
                'class': predicted_class_val,
                'confidence': confidence_val,
                'label': disease_label
            })
            
            # Show top 3 for first test
            if i == 0:
                top3_conf, top3_classes = torch.topk(probabilities, 3)
                print("Top 3 predictions:")
                for j in range(3):
                    class_idx = top3_classes[j].item()
                    conf = top3_conf[j].item()
                    label = id2label.get(str(class_idx), f"Class_{class_idx}")
                    print(f"  {j+1}. Class {class_idx} ({label}): {conf:.6f}")
    
    # Analysis
    print("\n" + "=" * 50)
    print("📊 ANALYSIS")
    print("=" * 50)
    
    unique_classes = set(r['class'] for r in results)
    unique_labels = set(r['label'] for r in results)
    unique_confidences = set(round(r['confidence'], 6) for r in results)
    
    print(f"Unique predicted classes: {len(unique_classes)} - {sorted(unique_classes)}")
    print(f"Unique labels: {len(unique_labels)} - {list(unique_labels)}")
    print(f"Unique confidence values: {len(unique_confidences)}")
    print(f"Confidence range: {min(r['confidence'] for r in results):.6f} - {max(r['confidence'] for r in results):.6f}")
    
    # Check for diversity
    if len(unique_classes) > 3:
        print("✅ GOOD: Model produces diverse predictions for different inputs")
    elif len(unique_classes) > 1:
        print("⚠️ OKAY: Some diversity in predictions")
    else:
        print("❌ PROBLEM: Model always predicts the same class")
        
    if len(unique_confidences) > 3:
        print("✅ GOOD: Confidence values vary with input")
    else:
        print("❌ PROBLEM: Confidence values are too similar")

if __name__ == "__main__":
    test_model_directly()