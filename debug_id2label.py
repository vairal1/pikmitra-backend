#!/usr/bin/env python3
"""
Debug id2label mapping
"""

from transformers import AutoModelForImageClassification

def debug_id2label():
    model_path = '../model/mobilenet_v2_1.0_224-plant-disease-identification'
    model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
    
    print("ID2Label mapping (first 10):")
    for i, (key, value) in enumerate(model.config.id2label.items()):
        print(f"  Key: '{key}' (type: {type(key)}) -> Value: '{value}'")
        if i >= 9:
            break
    
    # Test predictions for some indices
    test_indices = [0, 2, 4, 8, 10, 19, 30]
    print(f"\nTesting label lookup for indices: {test_indices}")
    for idx in test_indices:
        str_key = str(idx)
        if str_key in model.config.id2label:
            print(f"  Index {idx} -> '{model.config.id2label[str_key]}'")
        else:
            print(f"  Index {idx} -> NOT FOUND in id2label")

if __name__ == "__main__":
    debug_id2label()