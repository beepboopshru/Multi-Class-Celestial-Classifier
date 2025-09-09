# 🌌 Multi-Class Celestial Classifier

*A work-in-progress deep learning project to classify celestial bodies — Nebulas, Stars, and Galaxies.*

---

## 📖 Project Overview

This project is a **multi-class image classifier** designed to categorize telescope images of celestial bodies into **three classes**:

* 🌠 **Stars**
* 🌌 **Galaxies**
* ☁️ **Nebulas**

The model is being built using **Convolutional Neural Networks (CNNs)** and serves as a step toward applying machine learning to **astronomical image analysis**.

Currently, the project is **in progress** — datasets are being curated, and different CNN architectures are being experimented with for optimal accuracy.

---

## 🚀 Features (Planned & In Progress)

* [x] Image preprocessing (normalization, resizing, augmentation)
* [x] Dataset split into train/validation/test
* [ ] Baseline CNN model training
* [ ] Experimentation with transfer learning (ResNet, VGG, EfficientNet)
* [ ] Performance evaluation with accuracy, F1-score, and confusion matrix
* [ ] Interactive visualization of predictions

---

## 📂 Project Structure

```bash
celestial-classifier/
│── data/                # Dataset (images of nebulas, stars, galaxies)
│── notebooks/           # Jupyter notebooks for experiments
│── src/                 # Core source code
│   ├── dataset.py       # Data loading and preprocessing
│   ├── model.py         # CNN / Transfer Learning models
│   ├── train.py         # Training loop and validation
│   ├── evaluate.py      # Model evaluation metrics
│── results/             # Model outputs (plots, logs, checkpoints)
│── requirements.txt     # Project dependencies
│── README.md            # Project documentation
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/celestial-classifier.git
cd celestial-classifier
pip install -r requirements.txt
```

---

## 🖥️ Usage

### 1. Train the model

```bash
python src/train.py --epochs 20 --batch_size 32
```

### 2. Evaluate the model

```bash
python src/evaluate.py --model checkpoints/model_best.pth
```

### 3. Run predictions on new images

```bash
python src/predict.py --image path/to/your/image.jpg
```

---

## 📊 Current Progress

* ✅ Dataset collection: \~5000+ labeled images
* ✅ Initial preprocessing pipeline completed
* 🔄 Baseline CNN model training underway
* 🔜 Experimentation with transfer learning

---

## 🔭 Future Work

* Expand dataset with more diverse astronomical images
* Implement real-time inference for large datasets
* Integrate Grad-CAM for explainable AI in astronomy
* Deploy model with a simple web interface

---

## 🧑‍💻 Contributing

This is a **learning and exploration project**. Contributions, feedback, and discussions are welcome. Open a pull request or issue if you’d like to help improve the project!

---

## 📜 License

This project is licensed under the **MIT License**.

---
