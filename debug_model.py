#!/usr/bin/env python3
"""
Debug model inference to check if it's always returning the same prediction
"""

import sys
sys.path.append('.')

from app import *
import torch
import numpy as np

def test_model_inference():
    """Test model with different random inputs"""
    print("🔬 Debugging Model Inference")
    print("=" * 50)
    
    # Load model
    load_model()
    load_cures()
    
    print(f"Model loaded: {model is not None}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"Disease classes loaded: {len(disease_classes)}")
    
    if model is None:
        print("❌ Model not loaded!")
        return
    
    print("\n🧪 Testing with different random inputs...")
    
    # Test with 5 different random tensors
    for i in range(5):
        print(f"\n--- Test {i+1} ---")
        
        # Create random input tensor
        img_tensor = torch.randn(1, 3, 224, 224)
        
        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            print(f"Output shape: {logits.shape}")
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(logits[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
            
            confidence_val = confidence.item()
            predicted_class_val = int(predicted_class.item())
            
            print(f"Predicted class: {predicted_class_val}")
            print(f"Confidence: {confidence_val:.6f}")
            
            # Get top 3 predictions
            top3_conf, top3_classes = torch.topk(probabilities, 3)
            print("Top 3 predictions:")
            for j in range(3):
                class_idx = top3_classes[j].item()
                conf = top3_conf[j].item()
                label = disease_classes.get(str(class_idx), f"Class_{class_idx}")
                print(f"  {j+1}. Class {class_idx} ({label}): {conf:.6f}")
    
    print("\n" + "=" * 50)
    print("🎯 Analysis:")
    print("If all tests show the same class with identical confidence,")
    print("there might be an issue with model loading or inference.")

if __name__ == "__main__":
    test_model_inference()