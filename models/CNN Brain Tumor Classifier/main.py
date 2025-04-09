
import torch
import argparse
from train import train_model
from inference import load_model, predict_single_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='C:/Users/jason\Desktop/bounding box predictor copy/models/CNN Brain Tumor Classifier/data/images', help='Path to images folder')
    parser.add_argument('--labels_dir', type=str, default='C:/Users/jason\Desktop/bounding box predictor copy/models/CNN Brain Tumor Classifier/data/labels', help='Path to labels folder')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--save_model', type=str, default='C:/Users/jason\Desktop/bounding box predictor copy/models/CNN Brain Tumor Classifier/tumor_classifier.pth', help='Where to save trained model')
    parser.add_argument('--inference_image', type=str, default='', help='Path to image for inference')
    args = parser.parse_args()
    
    # Train the model
    model = train_model(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )
    
    # Save model
    torch.save(model.state_dict(), args.save_model)
    print(f"Model saved to {args.save_model}")
    
    # If inference_image is provided, do inference
    if args.inference_image:
        model.eval()
        pred, prob = predict_single_image(model, args.inference_image)
        label_str = "Tumor" if pred == 1 else "No Tumor"
        print(f"Prediction: {label_str} (prob={prob:.4f})")

if __name__ == '__main__':
    main()
