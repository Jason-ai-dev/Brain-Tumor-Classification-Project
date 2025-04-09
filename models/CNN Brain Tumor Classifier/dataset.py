import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    """
    Custom dataset for brain tumor detection.
    Expects two folders:
        - images_dir: images of MRI/CT scans
        - labels_dir: corresponding .txt files with labels in the format:
            1 0.344484 0.342723 0.221831 0.176056
          The first value is the class label (0 or 1). The rest are bounding box coords (ignored).
    """
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # List all image file paths
        self.image_paths = sorted([
            os.path.join(images_dir, fname) 
            for fname in os.listdir(images_dir) 
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Construct label path by matching the image base name
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, base_name + ".txt")
        
        # Load and preprocess the image
        image = Image.open(img_path).convert("L")  # ensure single-channel grayscale
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Read the label (first value in the text file)
        with open(label_path, 'r') as f:
            line = f.readline().strip()
        label = int(line.split()[0])  # the classification label (0 or 1)
        
        return image, label
