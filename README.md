# SQL_Injection

--

# 🛡️ SQL Injection Detection Using Logistic Regression

This project demonstrates how to build a **machine learning model** using **Logistic Regression** to detect **SQL injection attacks**. It uses labeled training and testing data to evaluate the performance of the model through metrics like **precision**, **recall**, **F1-score**, **MAE**, and **AUC-ROC**.

## 📌 Features

* Logistic Regression for binary classification (malicious or benign)
* Evaluation using:

  * Classification Report
  * Mean Absolute Error (MAE)
  * ROC Curve and AUC Score
* Visualization of the ROC Curve for performance interpretation

---

## 🚀 Getting Started

### Prerequisites

* Python 3.7+
* scikit-learn
* matplotlib
* numpy
* pandas (if loading data from CSV)

Install required libraries:

```bash
pip install scikit-learn matplotlib numpy pandas
```

### Dataset

The dataset should be preprocessed and split into:

* `X_train`, `y_train` — Training features and labels
* `X_test`, `y_test` — Testing features and labels

---

## 🧠 Model Training

```python
from sklearn.linear_model import LogisticRegression

# Initialize and train
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict
y_pred = log_reg.predict(X_test)
```

---

## 📊 Evaluation

### Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

**Sample Output:**

```
              precision    recall  f1-score   support

           0       0.88      0.98      0.93      2271
           1       0.97      0.87      0.92      2282

    accuracy                           0.92      4553
   macro avg       0.93      0.92      0.92      4553
weighted avg       0.93      0.92      0.92      4553
```

### Mean Absolute Error (MAE) and ROC Curve

```python
from sklearn.metrics import mean_absolute_error, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

try:
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC: {auc_roc}")

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_roc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
except ValueError:
    print("Could not calculate AUC-ROC. Check if y_pred contains only 0 and 1.")
```

---

## 📁 Project Structure

```
sql-injection-detector/
│
├── data/                    # Dataset files (if any)
├── models/                  # Trained models (optional)
├── src/
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Evaluation script
│
├── README.md
└── requirements.txt
```

---

## ✅ Results

* **Accuracy**: \~92%
* **AUC-ROC**: \~0.92
* **MAE**: \~0.08

These results suggest a well-performing binary classification model for identifying potential SQL injection attacks.

---

## 📌 Future Improvements

* Use advanced ML models (e.g., Random Forest, XGBoost)
* Perform feature selection or engineering
* Test with real-world SQLi payloads
* Integrate with web application firewalls (WAFs)

---
