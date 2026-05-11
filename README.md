# 🎭 NLP Emotions Classification

> Classify English text into 6 emotions using Machine Learning & Deep Learning (LSTM)

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/dair-ai/emotion)
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-222?logo=github&logoColor=white)](https://username.github.io/emotion-classifier/)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Emotions](#-emotions)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline](#-pipeline)
- [Models & Results](#-models--results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Saved Artefacts](#-saved-artefacts)

---

## 🧠 Overview

This project builds an end-to-end **text emotion classification** system that detects one of six emotions from raw English text. Two parallel pipelines were implemented and compared:

| Pipeline | Approach | Best Accuracy |
|----------|----------|--------------|
| **ML** | TF-IDF + 6 Classifiers | ~89% (Logistic Regression) |
| **Deep Learning** | LSTM Neural Network | ~92% |

---


## 🎭 Emotions

The model classifies text into **6 emotion classes**:

| Label | Emotion | Emoji |
|-------|---------|-------|
| 0 | Sadness | 😔 |
| 1 | Joy | 😄 |
| 2 | Love | ❤️ |
| 3 | Anger | 😡 |
| 4 | Fear | 😨 |
| 5 | Surprise | 😲 |

---

## 📦 Dataset

**Source:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) on Hugging Face

| Split | Samples |
|-------|---------|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

**Text stats:** avg length ≈ 67 chars · max ≈ 236 · min ≈ 7

```python
from datasets import load_dataset
dataset = load_dataset('parquet', data_files={
    'train':      'train-00000-of-00001.parquet',
    'validation': 'validation-00000-of-00001.parquet',
    'test':       'test-00000-of-00001.parquet'
})
```

---

## 📁 Project Structure

```
emotion-classifier/
│
├── Nlp_Emotions_Classification_V1.ipynb   # Main notebook
│
├── data/
│   ├── train-00000-of-00001.parquet
│   ├── validation-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
│
├── models/                                # Saved ML models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── svm_model.pkl
│   ├── gradient_boosting.pkl
│   ├── multinomial_nb.pkl
│   ├── knn_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── id2label.pkl
│
├── deep_model/                            # Saved LSTM model
│   ├── deep_bilstm_emotion.keras
│   ├── lstm_tokenizer.pkl
│   └── id2label.pkl
│
├── deployment/                            # GitHub Pages website
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── README.md
```

---

## 🔄 Pipeline

```
Raw Text
   │
   ▼
Text Cleaning
  ├─ Remove non-alpha chars (regex)
  ├─ Lowercase
  ├─ Remove punctuation
  ├─ Tokenize (NLTK)
  ├─ Remove stop words
  └─ Lemmatize (WordNet)
   │
   ├──────────────────────────────────────┐
   ▼                                      ▼
TF-IDF Vectorizer                  Keras Tokenizer + Padding
(max 5000 features, ngrams 1-3)    (vocab=20k, maxlen=100)
   │                                      │
   ▼                                      ▼
ML Classifiers (×6)              LSTM Neural Network
   │                                      │
   ▼                                      ▼
Predicted Emotion              Predicted Emotion + Confidence
```

---

## 📊 Models & Results

### Machine Learning Classifiers

| Model | Test Accuracy | F1 (Weighted) | Train Time |
|-------|:-------------:|:-------------:|:----------:|
| **Logistic Regression** ⭐ | ~0.89 | ~0.89 | ~2s |
| SGD / SVM | ~0.87 | ~0.87 | ~3s |
| Random Forest | ~0.84 | ~0.84 | ~30s |
| Multinomial Naive Bayes | ~0.83 | ~0.83 | ~0.1s |
| Gradient Boosting | ~0.82 | ~0.82 | ~90s |
| K-Nearest Neighbors | ~0.78 | ~0.78 | ~1s |

### Deep Learning — LSTM

| Metric | Value |
|--------|-------|
| Test Accuracy | **~0.92** |
| F1 Weighted | **~0.92** |
| F1 Macro | **~0.91** |

**LSTM Architecture:**
```
Embedding(20000, 128, input_length=100)
  → SpatialDropout1D(0.3)
  → LSTM(128, dropout=0.3, recurrent_dropout=0.2)
  → BatchNormalization()
  → Dense(128, relu)
  → Dropout(0.5)
  → Dense(6, softmax)
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/username/emotion-classifier.git
cd emotion-classifier

# Install dependencies
pip install datasets scikit-learn nltk tensorflow seaborn matplotlib wordcloud

# Download NLTK data
python -c "import nltk; nltk.download(['stopwords','punkt','wordnet','omw-1.4','punkt_tab'])"
```

---

## 🚀 Usage

### Run the notebook
```bash
jupyter notebook Nlp_Emotions_Classification_V1.ipynb
```

### Predict with ML model
```python
import pickle

# Load model & vectorizer
model  = pickle.load(open("models/logistic_regression.pkl", "rb"))
tfidf  = pickle.load(open("models/tfidf_vectorizer.pkl",   "rb"))
labels = pickle.load(open("models/id2label.pkl",           "rb"))

def predict(text):
    vec  = tfidf.transform([text])
    lid  = model.predict(vec)[0]
    return labels[lid]

print(predict("I feel so happy today!"))   # → joy
```

### Predict with LSTM model
```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model     = load_model("deep_model/deep_bilstm_emotion.keras")
tokenizer = pickle.load(open("deep_model/lstm_tokenizer.pkl", "rb"))
labels    = pickle.load(open("deep_model/id2label.pkl",       "rb"))

def predict_lstm(text, max_len=100):
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='pre')
    probs  = model.predict(padded, verbose=0)[0]
    lid    = int(np.argmax(probs))
    return labels[lid], float(probs[lid])

emotion, confidence = predict_lstm("I feel so alone")
print(f"{emotion} ({confidence:.1%})")    # → sadness (94.3%)
```

---

## 🌐 Deployment

The project is deployed as a **static website on GitHub Pages**.

### Live Demo

🔗 **[ live Demo](https://noga-66.github.io/Text-Emotion-classification/)**

### Deploy Steps

1. Create a GitHub repository named `emotion-classifier`
2. Upload all files (`index.html`, `style.css`, `app.js`, model files)
3. Go to **Settings → Pages**
4. Under *Source*, select **main branch** → **root folder**
5. Click **Save** — site goes live automatically

### Website Features
- 📝 Text input area for any English sentence
- 🎯 Predicted emotion with emoji and confidence score
- 📊 Probability bar chart for all 6 emotions
- 💡 Pre-loaded example sentences
- 📱 Mobile-friendly responsive design

---

## 💾 Saved Artefacts

| File | Description |
|------|-------------|
| `logistic_regression.pkl` | Tuned Logistic Regression |
| `random_forest.pkl` | Random Forest (300 trees) |
| `svm_model.pkl` | SGD / SVM (hinge loss) |
| `gradient_boosting.pkl` | Gradient Boosting (200 estimators) |
| `multinomial_nb.pkl` | Multinomial Naive Bayes (α=0.1) |
| `knn_model.pkl` | K-Nearest Neighbors (k=7, cosine) |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `id2label.pkl` | Label mapping dictionary |
| `deep_bilstm_emotion.keras` | Full LSTM model with weights |
| `lstm_tokenizer.pkl` | Fitted Keras tokenizer |

---


---

<p align="center">Made with ❤️ — Nada Hossam 2026</p>
