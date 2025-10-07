# 🔥 Wildfire Prediction from Satellite Imagery using Deep Learning 🌍

**Accurate wildfire detection from satellite data using Convolutional Neural Networks (CNNs)**
*Achieved an impressive 95.3% accuracy — demonstrating the power of AI for environmental monitoring.*

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Why Use a CNN?](#why-use-a-cnn)
3. [The Development Environment: Why Jupyter Notebooks?](#the-development-environment-why-jupyter-notebooks)
4. [Dataset](#dataset)
5. [Model Architecture and Implementation](#model-architecture-and-implementation)
6. [Results and Evaluation](#results-and-evaluation)
7. [Societal Impact](#societal-impact)
8. [How to Run the Project Locally](#how-to-run-the-project-locally)
9. [Project Structure](#project-structure)
10. [Conclusion and Future Work](#conclusion-and-future-work)

---

## 🌲 Project Overview

Wildfires are a growing environmental threat, destroying ecosystems and communities worldwide. Early and accurate detection is **crucial** for effective response and management.

This project leverages **deep learning** to automatically classify satellite images as **'Fire'** or **'No_Fire'**.
The model processes 224x224 pixel imagery and achieves **95.3% accuracy**, serving as a proof-of-concept for automated, scalable environmental monitoring.

---

## 🧠 Why Use a CNN?

Convolutional Neural Networks (CNNs) are the gold standard for image-based tasks such as wildfire detection.
Unlike traditional ML models, CNNs **automatically learn spatial and contextual features** from image data.

### Key Advantages

* **Spatial Hierarchy:** Learns simple textures → smoke patterns → full fire recognition.
* **Feature Extraction:** Automatically identifies smoke, flames, or healthy vegetation anywhere in the image.
* **Efficiency:** Optimized for handling large volumes of image data from satellites.

---

## 💻 The Development Environment: Why Jupyter Notebooks?

Jupyter Notebooks were chosen as the development environment due to their:

* **Interactive Workflow:** Run code cell-by-cell for iterative testing and debugging.
* **Inline Visualizations:** Instantly visualize images, metrics, and plots.
* **Narrative Storytelling:** Combine code, text, and visuals into one shareable, reproducible document.

---

## 🛰️ Dataset

The model uses the **Wildfire Prediction Dataset** from Kaggle, derived from **Canadian forest satellite imagery** captured via the MapBox API.

* **Source:** [Wildfire Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
* **Original Data:** Canadian National Fire Database
* **Content:** 350×350 pixel RGB satellite images

### Data Split

| Set            | Purpose                    | Distribution |
| :------------- | :------------------------- | :----------: |
| **Training**   | Train model weights        |     ~70%     |
| **Validation** | Tune hyperparameters       |     ~15%     |
| **Testing**    | Evaluate final performance |     ~15%     |

```
Dataset/
├── train/
│   ├── Fire/
│   └── No_Fire/
├── valid/
│   ├── Fire/
│   └── No_Fire/
└── test/
    ├── Fire/
    └── No_Fire/
```

---

## 🧩 Model Architecture and Implementation

This custom CNN is implemented using **TensorFlow Keras**.
The architecture was carefully optimized for accuracy, efficiency, and compatibility with TensorFlow 2.x.

### Key Components

1. **Data Preprocessing**

   * `ImageDataGenerator` for loading and augmenting images (rescaling, resizing to 224×224).

2. **Model Layers**

   * `Conv2D` + `ReLU` activations → spatial feature extraction.
   * `MaxPool2D` → downsampling.
   * `BatchNormalization` → stabilizes training.
   * `GlobalAveragePooling2D` → aggregates learned features.
   * `Dense(softmax)` → outputs class probabilities (‘Fire’, ‘No_Fire’).

3. **Training Setup**

   * **Optimizer:** Adam
   * **Loss Function:** Categorical Crossentropy
   * **Metric:** Accuracy
   * **Callback:** `ModelCheckpoint` saves best-performing model weights.

---

## 📈 Results and Evaluation

* **Final Test Accuracy:** 95.3%
* The model generalizes well, showing minimal overfitting and high discriminative performance.
* Training and validation plots confirmed steady convergence and learning consistency.

---

## 🌍 Societal Impact

This AI-powered wildfire detection system offers immense real-world value:

* **Early Warning:** Detects fires from satellite imagery long before human reports.
* **Resource Optimization:** Enables better allocation of firefighting assets.
* **Protection:** Safeguards lives, homes, and wildlife.
* **Climate Mitigation:** Helps reduce carbon release from large wildfires.
* **Economic Benefit:** Reduces financial losses due to property and environmental damage.

---

## ⚙️ How to Run the Project Locally

### Prerequisites

* Git
* Python 3.8+

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/wildfire-prediction-project.git
   cd wildfire-prediction-project
   ```

2. **Create a Virtual Environment**

   ```bash
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**
   Organize the dataset as shown in the [Dataset section](#dataset).

5. **Run the Notebook**

   ```bash
   jupyter lab
   ```

   Open and run `3_model_with_cam.ipynb` sequentially.

---

## 🗂️ Project Structure

```
├── Dataset/
│   ├── train/
│   ├── valid/
│   └── test/
├── saved_model/
│   └── custom_model.keras
├── predictions/
│   └── custom_model_predictions.csv
├── 1_load_and_display_data.ipynb
├── 2_pretrained_vgg16.ipynb
├── 3_model_with_cam.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Conclusion and Future Work

This project demonstrates how **CNNs** can effectively detect wildfires from satellite imagery with high accuracy and reliability.

### Future Enhancements

* Integrate **meteorological data** (wind, humidity, temperature) for richer predictions.
* Experiment with **ResNet** or **EfficientNet** architectures.
* Implement **real-time detection** pipelines for deployment in active wildfire monitoring systems.

---

**🌎 Built with Deep Learning to Protect the Planet 🔥**
*“Project by two Awesome People - Anshveer Turna and Harsh Yadav”*

---

