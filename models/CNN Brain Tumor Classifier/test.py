import os
import shutil
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import random
from sklearn.metrics import precision_score, recall_score

from dataset import BrainTumorDataset
from transforms import get_transforms
from model import TumorClassifierCNN

def test_model(images_dir, labels_dir, model_path,
               batch_size=32, random_seed=1234, threshold=0.5):
    """
    Loads the saved model and evaluates it on the 15% test subset.
    Saves all predicted-positive images and their original labels
    from the dataset into images/ and labels/.
    """
    # Fix random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Load dataset and split into train/val/test
    transform = get_transforms()
    dataset = BrainTumorDataset(images_dir, labels_dir, transform=transform)
    total_length = len(dataset)
    train_size = int(0.70 * total_length)
    val_size = int(0.15 * total_length)
    test_size = total_length - train_size - val_size

    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader for the test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the trained model
    model = TumorClassifierCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Evaluate the model
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).long()

            correct += (preds.squeeze() == labels.squeeze().long()).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.squeeze().cpu().numpy().tolist())
            all_preds.extend(preds.squeeze().cpu().numpy().tolist())

            # Save all predicted-positive images and their labels
            for i in range(images.size(0)):
                if preds[i].item() == 1:  # Tumor detected
                    # Get the original file path and label
                    orig_idx = test_dataset.indices[batch_idx * batch_size + i]
                    src_image_path = dataset.image_paths[orig_idx]
                    
                    # Construct the label path based on the dataset's convention
                    src_label_path = os.path.join(labels_dir, os.path.basename(src_image_path).replace('.jpg', '.txt').replace('.png', '.txt'))

                    # Check if the label file exists
                    if not os.path.exists(src_label_path):
                        print(f"Warning: Label file not found for {src_image_path}")
                        continue

                    # Save the image
                    dst_image_path = os.path.join('C:/Users/jason/Desktop/bounding box predictor copy/data/yolo_dataset/test/images', os.path.basename(src_image_path))
                    shutil.copy(src_image_path, dst_image_path)

                    # Save the label
                    dst_label_path = os.path.join('C:/Users/jason/Desktop/bounding box predictor copy/data/yolo_dataset/test/labels', os.path.basename(src_label_path))
                    shutil.copy(src_label_path, dst_label_path)

    # Calculate metrics
    avg_loss = total_loss / total
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print(f"Test Loss: {avg_loss:.4f}, "
          f"Accuracy: {accuracy*100:.2f}% on {total} samples")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

def main():
    images_dir = "C:/Users/jason\Desktop/bounding box predictor copy/models/CNN Brain Tumor Classifier/data/images"  # Replace with the actual path to images
    labels_dir = "C:/Users/jason\Desktop/bounding box predictor copy/models/CNN Brain Tumor Classifier/data/labels"  # Replace with the actual path to labels
    model_path = "C:/Users/jason\Desktop/bounding box predictor copy/models/CNN Brain Tumor Classifier/tumor_classifier.pth"  # Replace with the actual model path
    batch_size = 32
    random_seed = 1234
    threshold = 0.2  # Adjust this value as needed

    test_model(images_dir, labels_dir, model_path, batch_size, random_seed, threshold)

if __name__ == "__main__":
    main()