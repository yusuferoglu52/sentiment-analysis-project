# 🎬 Sentiment Analysis with Classical ML & BERT

This project performs **sentiment analysis on movie reviews** using both traditional machine learning models and a modern transformer-based model (**BERT**).

The goal is to classify reviews as **positive** or **negative** and compare different NLP approaches.

---

## 🚀 Features

* 🔤 Text preprocessing pipeline
* 📊 TF-IDF based feature extraction
* 🤖 Multiple models:

  * Logistic Regression
  * Linear SVM
  * BERT (Transformer)
* ⚖️ Model comparison & evaluation
* 📉 Confusion matrix visualization
* ❌ Error analysis (misclassified samples)
* 🧪 Interactive prediction script

---

## 📁 Project Structure

```
sentiment-analysis-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_baseline_tfidf_logreg.ipynb
│
├── src/
│   ├── train.py
│   ├── train_bert.py
│   ├── predict.py
│   ├── preprocess.py
│
├── models/
├── outputs/
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* IMDb Movie Reviews Dataset (50K samples)
* Balanced dataset (positive / negative)

👉 Download from:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## ⚙️ Models Used

### 🔹 Classical ML

* TF-IDF + Logistic Regression
* TF-IDF + Linear SVM

### 🔹 Transformer

* BERT (`bert-base-uncased`)

---

## 📈 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~0.91    |
| Linear SVM          | ~0.90    |
| BERT                | ~0.92+   |

### Key Observations

* Linear models provide strong baselines
* BERT significantly improves performance
* BERT handles:

  * negation ("not good")
  * mixed sentiment
  * contextual meaning

---

## 🧠 Example

Input:

```
The movie was not bad at all.
```

Output:

```
Logistic Regression: Negative ❌
Linear SVM: Negative ❌
BERT: Positive ✅
```

---

## 🧪 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

---

### 2. Train classical models

```
python src/train.py
```

---

### 3. Train BERT (recommended on GPU / Colab)

```
python src/train_bert.py
```

---

### 4. Run predictions

```
python src/predict.py
```

---

## 📉 Error Analysis

The models struggle with:

* negation
* sarcasm
* mixed sentiment sentences

BERT performs better due to contextual understanding.

---

## 📊 Visualizations

* Confusion matrix plots
* Model comparison reports
* Misclassified samples

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Larger transformer models
* Model deployment (API)
* Real-time sentiment analysis

---

## 👤 Author

yusuferoglu52

---

## ⭐ Notes

* Large files (data, models) are excluded via `.gitignore`
* BERT model should be trained separately (preferably with GPU)

---

## 🧠 Takeaway

This project demonstrates the transition from:

* traditional NLP → modern transformer-based NLP

and highlights the trade-off between:

* speed vs performance
