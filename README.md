# Brain Tumor Finder

A two‑stage brain tumor detection system that first classifies MRI scans as “tumor” vs. “no tumor” using a CNN, then pinpoints tumor locations with a YOLOv8 object‑detector.

---

## 📂 Directory Structure

Brain Tumor Finder/ ├── data/ │ └── yolo_dataset/ # YOLO‑formatted images + labels ├── models/ │ └── CNN Brain Tumor Classifier/ │ ├── dataset.py # Data loading & preprocessing │ ├── inference.py # Single‑image inference utilities │ ├── main.py # Training entry point │ ├── model.py # CNN architecture │ ├── test.py # Batch testing & metrics │ ├── train.py # Training loops & helpers │ └── transforms.py # TorchVision transforms ├── runs/ # Training logs & checkpoints ├── results/ │ ├── predictions/ # YOLO output images (bbox) │ └── visualizations/ # Side‑by‑side ground truth vs. pred ├── src/ │ ├── config/ │ │ └── config.py # All hyperparameters & paths │ └── models/ │ ├── yolo.py # YOLOv8 model setup │ └── predict.py # Run YOLO inference on positives └── tumor_classifier.pth # Trained CNN weights

yaml
Copy
Edit

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch 1.10+
- Ultralytics YOLOv8
- OpenCV
- Pillow
- Matplotlib
- NumPy

```bash
pip install torch torchvision ultralytics opencv-python pillow matplotlib numpy
🚀 Usage
1. Train the CNN classifier
bash
Copy
Edit
cd models/"CNN Brain Tumor Classifier"
python main.py
Loads and augments your MRI dataset

Builds & trains the CNN

Saves tumor_classifier.pth to the project root

Prints training loss & accuracy

2. Evaluate the CNN
bash
Copy
Edit
cd models/"CNN Brain Tumor Classifier"
python test.py
Loads tumor_classifier.pth

Runs inference on your test set

Outputs per‑image labels (0 = no tumor, 1 = tumor)

Prints overall accuracy, precision, recall

Copies all “1”‑labeled images into a staging folder for YOLO

3. Localize tumors with YOLOv8
bash
Copy
Edit
cd src/models
python predict.py
Loads the pretrained YOLOv8 brain‑tumor model

Processes all CNN‑positive scans

Draws bounding boxes around tumors

Computes precision, recall, mAP

Saves results to results/predictions & results/visualizations

📊 Expected Outputs
Binary labels for each scan (0 or 1)

Bounding‑box images in results/predictions/

Side‑by‑side comparison in results/visualizations/

Metrics report (accuracy, precision, recall, mAP)

📝 Configuration
All paths, hyperparameters, and model settings live in:

arduino
Copy
Edit
src/config/config.py
Feel free to tweak learning rates, batch sizes, YOLO thresholds, etc., right there.

🔍 Notes
This two‑stage pipeline gives you both detection (is there a tumor?) and localization (where is it?).

For best results, ensure your MRI scans are pre‑aligned and normalized.

You can swap in your own dataset by following the same folder structure under data/ and adjusting config.py.
