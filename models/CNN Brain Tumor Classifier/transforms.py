
import torchvision.transforms as transforms

def get_transforms():
    """
    Returns the transformations for preprocessing images.
    Adjust the values (like resize dimensions or normalization)
    based on dataset requirements.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),         # Resize to 128x128
        transforms.ToTensor(),                 # Convert to Tensor (0-1 range)
        transforms.Normalize([0.5], [0.5])     # Normalize (mean=0.5, std=0.5 for 1 channel)
    ])
    return transform
