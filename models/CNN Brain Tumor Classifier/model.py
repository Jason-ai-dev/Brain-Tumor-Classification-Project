
import torch.nn as nn

class TumorClassifierCNN(nn.Module):
    """
    A simple CNN model for binary classification of brain tumor images.
    Input: 1-channel (grayscale) 128x128 image
    Output: Single logit indicating tumor presence (1) or absence (0).
    """
    def __init__(self):
        super(TumorClassifierCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output: 16 x 64 x 64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output: 32 x 32 x 32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # output: 64 x 16 x 16
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64*16*16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # single output for binary classification
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x  # raw logits
