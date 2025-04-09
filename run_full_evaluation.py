#!/usr/bin/env python3
"""
Run full IoU evaluation on the test set
"""

import os
import json
from pathlib import Path
from src.models.predict import load_model, calculate_bbox_metrics

def main():
    """Run full evaluation and print results"""
    print("Loading model...")
    model = load_model('../models/best.pt')
    
    # Get dataset configuration
    project_root = Path(__file__).resolve().parent
    data_yaml = project_root / "data" / "yolo_dataset" / "dataset.yaml"
    
    if not data_yaml.exists():
        print(f"Dataset config not found: {data_yaml}")
        return
    
    print(f"Running evaluation on test set using {data_yaml}...")
    metrics = calculate_bbox_metrics(model, data_yaml)
    
    print("\n===== EVALUATION RESULTS =====")
    print(f"Mean IoU (mAP50): {metrics['mean_IoU']:.4f}")
    print(f"mAP50: {metrics['mAP50']:.4f}")
    print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Save results
    output_dir = project_root / "results" / "full_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_dir / "bbox_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {output_dir / 'bbox_metrics.json'}")

if __name__ == "__main__":
    main() 