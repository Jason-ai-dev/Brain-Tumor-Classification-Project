#!/usr/bin/env python3
"""
Bounding Box Predictor - Main Script
-----------------------------------
This script provides an easy way to run the bounding box predictor functionality.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bbox_predictor')

def setup_environment():
    """Set up environment for running the predictor"""
    # Add src directory to path
    root_dir = Path(__file__).parent
    src_dir = root_dir / "src"
    sys.path.append(str(src_dir))
    
    # Create necessary directories
    results_dir = root_dir / "results"
    models_dir = root_dir / "models"
    
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    return root_dir

def run_training(args):
    """Run the training process"""
    # Add src directory to path
    root_dir = Path(__file__).parent
    src_dir = root_dir / "src"
    sys.path.append(str(src_dir))
    
    # Import only after adding to path
    from src.models.train import train_bbox_model, evaluate_bbox_model

    # Prepare arguments for training
    class TrainArgs:
        def __init__(self):
            self.model_size = args.model_size if hasattr(args, 'model_size') else 'm'
            self.epochs = args.epochs if hasattr(args, 'epochs') else 16
            self.batch_size = args.batch_size if hasattr(args, 'batch_size') else 16
            self.img_size = args.img_size if hasattr(args, 'img_size') else 640
            self.pretrained = args.pretrained if hasattr(args, 'pretrained') else False
            self.workers = args.workers if hasattr(args, 'workers') else 8
            self.device = args.device if hasattr(args, 'device') else ''
            self.name = 'bbox_predictor'
            self.patience = args.patience if hasattr(args, 'patience') else 20
            self.verbose = args.verbose if hasattr(args, 'verbose') else False
    
    train_args = TrainArgs()
    
    # Run training
    logger.info(f"Starting training with model size {train_args.model_size}, {train_args.epochs} epochs")
    best_model_path = train_bbox_model(train_args)
    
    if best_model_path:
        logger.info(f"Training completed. Best model: {best_model_path}")
        
        # Get dataset yaml path for evaluation
        from src.models.yolo import get_project_root
        project_root = get_project_root()
        data_yaml = project_root / "data" / "yolo_dataset" / "dataset.yaml"
        
        # Run evaluation
        if data_yaml.exists():
            metrics = evaluate_bbox_model(best_model_path, data_yaml, train_args)
            logger.info("Evaluation metrics:")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}")
        else:
            logger.warning(f"Dataset YAML not found at {data_yaml}, skipping evaluation")
    else:
        logger.error("Training failed or best model not found")

def run_prediction(args):
    """Run the prediction process"""
    # Add src directory to path
    root_dir = Path(__file__).parent
    src_dir = root_dir / "src"
    sys.path.append(str(src_dir))
    
    # Import only after adding to path
    from src.models.predict import load_model, predict_bounding_boxes, predict_folder
    
    # Determine model path
    model_path = args.model if hasattr(args, 'model') else None
    if model_path is None:
        # Try to find best model
        model_path = root_dir / "models" / "bbox_predictor_best.pt"
        if not model_path.exists():
            # Check results directory
            results_dir = root_dir / "results" / "bbox_predictor" / "train" / "weights" / "best.pt"
            if results_dir.exists():
                model_path = results_dir
            else:
                logger.error("No trained model found. Please train a model first or specify a model path.")
                return
    
    # Load model
    try:
        model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Set default output directory if not provided
    output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else str(root_dir / "results" / "predictions")
    
    # Set confidence threshold
    conf_threshold = args.conf if hasattr(args, 'conf') else 0.25
    
    # Process single image
    image_path = args.image if hasattr(args, 'image') else None
    input_dir = args.input_dir if hasattr(args, 'input_dir') else None
    
    if image_path:
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return
        
        output_path = os.path.join(output_dir, f"bbox_{os.path.basename(image_path)}")
        logger.info(f"Running prediction on {image_path}")
        
        _, predictions = predict_bounding_boxes(model, image_path, conf_threshold, output_path)
        
        logger.info(f"Prediction completed with {len(predictions)} detections")
        for i, pred in enumerate(predictions):
            logger.info(f"  {i+1}. {pred['class_name']} ({pred['confidence']:.2f}): {pred['box']}")
    
    # Process directory of images
    elif input_dir:
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return
        
        logger.info(f"Processing images in {input_dir}")
        results = predict_folder(model, input_dir, output_dir, conf_threshold)
        
        logger.info(f"Processed {len(results)} images")
    
    else:
        logger.error("Please specify either --image or --input-dir for prediction")

def run_evaluation(args):
    """Run the evaluation process"""
    # Add src directory to path
    root_dir = Path(__file__).parent
    src_dir = root_dir / "src"
    sys.path.append(str(src_dir))
    
    # Import only after adding to path
    from src.models.predict import load_model, calculate_bbox_metrics
    from src.models.yolo import get_project_root
    
    # Determine model path
    model_path = args.model if hasattr(args, 'model') else None
    if model_path is None:
        # Try to find best model
        model_path = root_dir / "models" / "bbox_predictor_best.pt"
        if not model_path.exists():
            # Check results directory
            results_dir = root_dir / "results" / "bbox_predictor" / "train" / "weights" / "best.pt"
            if results_dir.exists():
                model_path = results_dir
            else:
                logger.error("No trained model found. Please train a model first or specify a model path.")
                return
    
    # Load model
    try:
        model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Get dataset yaml path
    project_root = get_project_root()
    data_yaml = project_root / "data" / "yolo_dataset" / "dataset.yaml"
    
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found at {data_yaml}")
        return
    
    # Run evaluation
    logger.info(f"Evaluating model on test set using {data_yaml}")
    metrics = calculate_bbox_metrics(model, data_yaml)
    
    # Print metrics
    logger.info("Bounding box evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Set default output directory if not provided
    output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else str(root_dir / "results" / "evaluation")
    
    # Save metrics to file
    import json
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "bbox_metrics.json")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")

def main():
    """Main function"""
    # Set up environment
    setup_environment()
    
    # If no command-line args provided, show usage
    if len(sys.argv) < 2:
        print("Usage: python run_bbox_predictor.py [train|predict|evaluate] [options]")
        return
    
    # Get command from first argument
    command = sys.argv[1]
    
    # Create a simplified parser for remaining arguments
    parser = argparse.ArgumentParser(description='Run Bounding Box Predictor')
    
    # Common arguments
    parser.add_argument('--model', type=str, help='Path to model weights')
    parser.add_argument('--model-size', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                     help='YOLOv8 model size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=16, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--device', type=str, default='', help='Device to use')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Prediction arguments
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--input-dir', type=str, help='Directory containing images')
    parser.add_argument('--output-dir', type=str, help='Directory to save results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    # Parse remaining arguments
    args = parser.parse_args(sys.argv[2:])
    
    # Execute command
    if command == 'train':
        run_training(args)
    elif command == 'predict':
        run_prediction(args)
    elif command == 'evaluate':
        run_evaluation(args)
    else:
        logger.error(f"Unknown command: {command}")
        print("Usage: python run_bbox_predictor.py [train|predict|evaluate] [options]")

if __name__ == "__main__":
    main() 