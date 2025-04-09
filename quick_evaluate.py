#!/usr/bin/env python3
"""
Quick evaluation script using direct ultralytics model evaluation
"""

from ultralytics import YOLO
from pathlib import Path

def main():
    # Load model
    print("Loading model...")
    model = YOLO("../models/best.pt")
    
    # Get dataset path
    project_root = Path(__file__).resolve().parent
    data_yaml = project_root / "data" / "yolo_dataset" / "dataset.yaml"
    
    if not data_yaml.exists():
        print(f"Dataset config not found: {data_yaml}")
        return
    
    print(f"Running evaluation on test set using {data_yaml}...")
    
    # Run validation and print all metrics
    results = model.val(
        data=data_yaml,
        split='test',
        verbose=True,  # Show details
    )
    
    # Print metrics in a more readable format
    print("\n===== DETAILED METRICS =====")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    # Try to access more IoU related information
    print("\n===== ADDITIONAL METRICS =====")
    # Print all available attributes of results.box
    print("Available box metrics:")
    for attr in dir(results.box):
        if not attr.startswith('_'):  # Skip private attributes
            try:
                value = getattr(results.box, attr)
                # Check if it's a method or a property
                if not callable(value):
                    print(f"  {attr}: {value}")
            except:
                print(f"  {attr}: <error accessing>")
    
    # Print detailed information about results structure
    print("\nResults structure:")
    for key in dir(results):
        if not key.startswith('_'):
            print(f"  {key}")

if __name__ == "__main__":
    main() 