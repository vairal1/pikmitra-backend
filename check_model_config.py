#!/usr/bin/env python3
"""
Check model configuration and available labels
"""

from transformers import AutoModelForImageClassification, AutoImageProcessor
import json

def check_model_config():
    print("🔍 Checking Model Configuration")
    print("=" * 50)
    
    # Load model and processor
    model_path = '../model/mobilenet_v2_1.0_224-plant-disease-identification'
    
    model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    
    print(f"📊 Model config:")
    print(f"  - num_labels: {model.config.num_labels}")
    print(f"  - problem_type: {getattr(model.config, 'problem_type', 'Not set')}")
    
    # Check for label mappings
    print(f"\n🏷️ Label mappings:")
    if hasattr(model.config, 'id2label'):
        print(f"  - id2label available: {len(model.config.id2label)} entries")
        print(f"  - Sample entries: {dict(list(model.config.id2label.items())[:5])}")
    else:
        print("  - id2label: Not available")
    
    if hasattr(model.config, 'label2id'):
        print(f"  - label2id available: {len(model.config.label2id)} entries")
        print(f"  - Sample entries: {dict(list(model.config.label2id.items())[:5])}")
    else:
        print("  - label2id: Not available")
    
    # Check config files directly
    import os
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        print(f"\n📄 Config file exists: {config_path}")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print(f"Config keys: {list(config_data.keys())}")
        
        if 'id2label' in config_data:
            print(f"id2label in config: {len(config_data['id2label'])} entries")
            print(f"First few: {dict(list(config_data['id2label'].items())[:5])}")
        
        if 'label2id' in config_data:
            print(f"label2id in config: {len(config_data['label2id'])} entries")
    else:
        print(f"\n❌ Config file not found: {config_path}")
    
    # Check what files are in the model directory
    print(f"\n📁 Files in model directory:")
    model_files = os.listdir(model_path)
    for file in sorted(model_files):
        print(f"  - {file}")
    
    # Check processor config
    print(f"\n🔧 Processor config:")
    print(f"  - size: {getattr(processor, 'size', 'Not available')}")
    print(f"  - do_normalize: {getattr(processor, 'do_normalize', 'Not available')}")
    print(f"  - image_mean: {getattr(processor, 'image_mean', 'Not available')}")
    print(f"  - image_std: {getattr(processor, 'image_std', 'Not available')}")

if __name__ == "__main__":
    check_model_config()