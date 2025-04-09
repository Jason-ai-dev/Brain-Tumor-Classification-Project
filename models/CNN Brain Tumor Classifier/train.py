
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random

from dataset import BrainTumorDataset
from transforms import get_transforms
from model import TumorClassifierCNN

def train_model(images_dir, labels_dir, 
                batch_size=32, 
                learning_rate=0.001, 
                num_epochs=15,
                model_save_path='tumor_classifier.pth',
                random_seed=1234,
                threshold=0.05):
    """
    Trains a CNN model for brain tumor classification, splitting data 70/15/15
    for train/val/test, but only uses train+val here. The test subset remains
    untouched for final testing in test.py.

    :param images_dir: Directory containing MRI/CT images
    :param labels_dir: Directory containing corresponding label .txt files
    :param batch_size: Batch size for training
    :param learning_rate: Learning rate for optimizer
    :param num_epochs: Number of training epochs
    :param model_save_path: Where to save the trained model
    :param random_seed: Ensures reproducible data splitting
    :return: trained model
    """
    
    # Fix the random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset and transforms
    transform = get_transforms()
    dataset = BrainTumorDataset(images_dir, labels_dir, transform=transform)
    
    total_length = len(dataset)
    train_size = int(0.70 * total_length)
    val_size   = int(0.15 * total_length)
    test_size  = total_length - train_size - val_size  # ensures total = 100%
    
    # Split dataset into train, val, test => 70/15/15
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # We only use train_dataset and val_dataset here.
    # test_dataset is not touched in training script => for test.py
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TumorClassifierCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()  # float for BCE loss
            
            optimizer.zero_grad()
            outputs = model(images)
            # shape of outputs: [batch_size, 1], shape of labels: [batch_size]
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            # Accumulate stats
            running_loss += loss.item() * labels.size(0)
            # Accuracy
            preds = (torch.sigmoid(outputs) >= threshold).long()
            correct += (preds.squeeze() == labels.long()).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item() * labels.size(0)
                
                preds = (torch.sigmoid(outputs) >= threshold).long()
                val_correct += (preds.squeeze() == labels.long()).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc*100:.2f}% "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model
