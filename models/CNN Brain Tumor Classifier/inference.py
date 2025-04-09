
import torch
from PIL import Image
from transforms import get_transforms
from model import TumorClassifierCNN

def load_model(checkpoint_path, device=None):
    """
    Loads the model weights from a checkpoint file.
    :param checkpoint_path: Path to the saved .pth file
    :param device: torch.device('cuda'/'cpu'), if None, auto-detect
    :return: An instance of the loaded model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TumorClassifierCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_single_image(model, image_path, device=None):
    """
    Runs inference on a single image to predict tumor presence.
    
    :param model: Trained TumorClassifierCNN
    :param image_path: Path to the image file
    :param device: torch.device('cuda'/'cpu'), if None, auto-detect
    :return: predicted label (0 or 1), probability of tumor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess the image
    transform = get_transforms()
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(tensor)  # shape: [1, 1]
        prob = torch.sigmoid(output).item()  # scalar probability
        pred = 1 if prob >= 0.5 else 0
    
    return pred, prob
