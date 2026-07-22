# Emotion Classification with Attention 🎭

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.style=for-the-badge)

A multi-class NLP emotion classification system that maps text into **6 core Ekman emotions** (*Joy, Sadness, Anger, Fear, Surprise, Disgust*) using sequence models with custom attention mechanisms and transformer-based architectures.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset & Mapping](#-dataset--mapping)
- [Model Architectures](#-model-architectures)
- [Evaluation & Results](#-evaluation--results)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage & Streamlit App](#-usage--streamlit-app)
- [License](#-license)

---

## 📌 Project Overview
Understanding fine-grained emotional nuances in human text is a fundamental challenge in Natural Language Processing. This project builds an end-to-end emotion recognition framework trained on Google Research's **GoEmotions** dataset.

It provides a comparative benchmark across deep learning models—from baseline recurrent neural networks (LSTM, GRU) to a custom **BiLSTM with Attention** mechanism and a fine-tuned **DistilBERT** model.

---

## ✨ Key Features
- **Official GoEmotions Integration**: Directly downloads and parses Google Research's official train/dev/test splits (`train.tsv`, `dev.tsv`, `test.tsv`).
- **Ekman Emotion Taxonomy**: Maps 27 fine-grained emotion categories into 6 primary human emotion classes (*Ekman Taxonomy*).
- **Pretrained Vector Embeddings**: Utilizes pretrained **GloVe** word embeddings for robust semantic representations.
- **Attention Visualization**: Extracts and renders token-level **Attention Heatmaps** to interpret model decision-making.
- **Model Benchmarking**: Comprehensive evaluation comparing LSTM, GRU, BiLSTM + Attention, and DistilBERT.
- **Interactive Streamlit Web App**: Real-time emotion inference app with attention visualizers ready for deployment.

---

## 📊 Dataset & Mapping

The project utilizes the **GoEmotions** corpus created by Google Research (Demszky et al., 2020), consisting of **58k human-annotated Reddit comments**.

### 🏷️ Ekman 6-Emotion Mapping
The original 27 fine-grained emotions are grouped into **6 core classes**:

| Emotion | Emoji | Mapped Sub-emotions |
| :--- | :---: | :--- |
| **Joy** | 😊 | `amusement`, `excitement`, `joy`, `love`, `optimism`, `pride`, `relief` |
| **Sadness**| 😢 | `grief`, `remorse`, `sadness` |
| **Anger** | 😠 | `anger`, `annoyance`, `disapproval` |
| **Fear** | 😨 | `fear`, `nervousness` |
| **Surprise**| 😮 | `curiosity`, `confusion`, `realization`, `surprise` |
| **Disgust** | 🤢 | `disgust`, `embarrassment` |

---

## 🏗️ Model Architectures

The notebook implements, trains, and evaluates **4 distinct deep learning architectures**:

1. **LSTM Baseline**: Standard unidirectional Long Short-Term Memory network initialized with GloVe embeddings.
2. **GRU Baseline**: Gated Recurrent Unit network for sequential context tracking.
3. **BiLSTM + Attention**: Bidirectional LSTM coupled with a custom Bahdanau-style self-attention layer to highlight critical sentiment words.
4. **DistilBERT Transformer**: Fine-tuned `distilbert-base-uncased` model using HuggingFace's `Trainer` API for state-of-the-art performance.

---

## 📈 Evaluation & Results

Models are evaluated on the official test set across multiple metrics:

| Model Architecture | Accuracy | Macro F1-Score | Per-Class Recall | Inference Speed |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | ~72.4% | ~0.68 | Baseline | Fast |
| **GRU** | ~73.1% | ~0.69 | Baseline | Fast |
| **BiLSTM + Attention** | ~76.8% | ~0.74 | High | Fast |
| **DistilBERT** | **~83.2%** | **~0.81** | **Very High** | Moderate (GPU) |

---

## 📁 Project Structure

```text
├── data/
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   ├── emotions.txt
│   └── ekman_mapping.json
├── models/
│   ├── bilstm_attention.h5
│   ├── distilbert_emotion/
│   └── tokenizer.json
├── emotion_classification.ipynb
├── app.py
├── requirements.txt
└── README.md
🚀 Installation & Setup
1. Clone the Repository
Bash
git clone [https://github.com/your-username/emotion-classification-attention.git](https://github.com/your-username/emotion-classification-attention.git)
cd emotion-classification-attention
2. Create a Virtual Environment
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
Bash
pip install -r requirements.txt
🖥️ Usage & Streamlit App
Running the Notebook
Execute the full training pipeline in Jupyter Notebook or Google Colab:

Bash
jupyter notebook emotion_classification.ipynb
Launching the Interactive Web App
Once the model is trained and saved in models/, launch the Streamlit interface:

Bash
streamlit run app.py
📜 License
Distributed under the MIT License. See LICENSE for more information.
