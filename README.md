
Brain Tumor Finder
This project implements a comprehensive brain tumor detection system using a two-stage approach:

A CNN classifier to determine if a brain scan contains a tumor (binary classification: 0 or 1)
A YOLOv8 model to predict the exact location of tumors using bounding boxes
Directory Structure
CopyInsert
Brain Tumor Finder/
├── data/                  # Dataset directory
│   └── yolo_dataset/      # YOLO formatted dataset
├── models/                # Model weights and configurations
│   └── CNN Brain Tumor Classifier/  # CNN classification model
│       ├── dataset.py     # Dataset loading utilities
│       ├── inference.py   # Inference utilities
│       ├── main.py        # Main script to run the CNN model
│       ├── model.py       # CNN model architecture
│       ├── test.py        # Script to test the CNN model
│       ├── train.py       # Training functions
│       └── transforms.py  # Image transformations
├── results/               # Prediction results and visualizations
├── runs/                  # Training run outputs
├── src/                   # Source code
│   ├── config/            # Configuration settings
│   │   └── config.py      # Configuration parameters
│   └── models/            # Model implementation files
│       ├── predict.py     # Script for bounding box prediction
│       └── yolo.py        # YOLO model setup utilities
└── tumor_classifier.pth   # CNN classifier model weights
How to Use
Follow these steps in order to perform the complete brain tumor detection workflow:

Step 1: Train CNN Classification Model
First, train the CNN model to classify brain scans as having tumors (1) or not having tumors (0).

bash
CopyInsert
cd "models\CNN Brain Tumor Classifier"
python main.py
This script:

Loads and preprocesses the training dataset
Builds and trains the CNN classifier model
Saves the trained model weights
Displays training progress and accuracy metrics
Step 2: Test CNN Model on Dataset
Next, evaluate the CNN model on the test dataset to identify brain scans containing tumors:

bash
CopyInsert
cd "models\CNN Brain Tumor Classifier"
python test.py
This script:

Loads the trained CNN model
Runs inference on the test dataset
Classifies each scan as 0 (no tumor) or 1 (tumor present)
Outputs classification results and accuracy metrics
Prepares positive cases for bounding box prediction
Step 3: Run YOLO Bounding Box Predictor
Finally, for the images classified as having tumors, use the YOLOv8 model to detect the precise location of the tumors:

bash
CopyInsert in Terminal
python src/models/predict.py
The YOLO prediction script:

Loads the pre-trained YOLOv8 model optimized for brain tumor detection
Processes images that were classified as positive (containing tumors) by the CNN
Generates bounding boxes around detected tumor regions
Calculates prediction metrics (precision, recall, mAP)
Creates visualization outputs in the results directory
Expected Outputs
After completing all three steps, you'll have:

Classification Results: Binary classification of each brain scan (0 or 1)
Bounding Box Visualizations: Images with highlighted tumor regions
Performance Metrics: Accuracy, precision, recall, and mAP scores for the models
All output visualizations are saved to the results directory, with:

results/predictions: Contains brain scans with predicted bounding boxes
results/visualizations: Shows comparative visualizations of predictions vs. ground truth
Requirements
Python 3.8 or higher
PyTorch 1.10 or higher
Ultralytics YOLOv8
OpenCV
PIL
Matplotlib
NumPy
Install dependencies with:

bash
CopyInsert in Terminal
pip install torch torchvision ultralytics opencv-python pillow matplotlib numpy
Notes
The CNN model identifies if a tumor is present in the scan
The YOLOv8 model precisely locates the tumor with a bounding box
This two-stage approach provides both detection and localization
Configuration settings can be modified in src/config/config.py
