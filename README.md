# 💳 Fraud Transaction Detector using Machine Learning

An end-to-end Machine Learning project that detects fraudulent financial transactions using transaction behavior patterns. The project includes data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

🚀 Live Demo:
https://fraud-transaction-detector-debdut-nandy.streamlit.app/

---

# 📌 Project Overview

Fraud detection is a critical problem in financial systems where identifying suspicious transactions quickly can prevent financial losses.

This project builds a Machine Learning classification system that predicts whether a transaction is:

- ✅ Legitimate Transaction
- ⚠️ Fraudulent Transaction

The model analyzes transaction details such as transaction type, amount, sender balance changes, and receiver balance changes to identify fraud patterns.

---

# 🎯 Objectives

- Analyze transaction behavior patterns
- Build a fraud classification model
- Handle imbalanced fraud datasets
- Perform feature engineering
- Evaluate model performance
- Deploy an interactive fraud detection application

---

# 🛠️ Tech Stack

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Machine Learning Algorithm

- Random Forest Classifier

---

# 📂 Project Structure

```

Fraud-Detection-System/

│

├── models/

│   ├── fraud_model.pkl

│   ├── scaler.pkl

│   ├── type_encoder.pkl

│   └── feature_columns.pkl

│

├── app.py

├── requirements.txt

├── README.md


```

---

# 📊 Dataset Features

The model uses transaction-level information:

| Feature | Description |
|-|-|
| step | Time step of transaction |
| type | Transaction category |
| amount | Transaction amount |
| oldbalanceOrg | Sender balance before transaction |
| newbalanceOrig | Sender balance after transaction |
| oldbalanceDest | Receiver balance before transaction |
| newbalanceDest | Receiver balance after transaction |
| balance_change_org | Sender balance difference |
| balance_change_dest | Receiver balance difference |

Target:

```

isFraud

0 → Legitimate Transaction

1 → Fraud Transaction

```

---

# ⚙️ Machine Learning Pipeline

## 1. Data Loading

- Imported transaction dataset
- Checked structure and statistics

---

## 2. Data Preprocessing

Performed:

- Removed unnecessary identifier columns
- Encoded categorical transaction types
- Created new balance-related features

---

## 3. Feature Engineering

Created:

### Sender Balance Change

```

oldbalanceOrg - newbalanceOrig

```

### Receiver Balance Change

```

oldbalanceDest - newbalanceDest

```

These features help identify abnormal money movement.

---

## 4. Model Training

Algorithm:

```

Random Forest Classifier

```

Configuration:

- Multiple decision trees
- Balanced class weights
- Optimized for fraud detection

---

# 📈 Model Performance

Evaluation metrics:

|    Metric      |  Score |
|----------------|--------|
| Fraud Recall   |   ~83% |
| Fraud F1 Score |   ~76% |

---

# 🖥️ Streamlit Application

The deployed application allows users to enter:

- Transaction Type
- Transaction Amount
- Sender Balance
- Receiver Balance
- Transaction Step

The ML model processes the input and returns:

### Prediction

```

✅ Legitimate Transaction

or

⚠️ Fraud Transaction Detected

```

along with fraud probability.

---

# 🔄 Application Workflow

```

User Input

```
  ↓
```

Feature Engineering

```
  ↓
```

Data Transformation

```
  ↓
```

Random Forest Model

```
  ↓
```

Fraud Probability Prediction

```
  ↓
```

Result Display

````

---

# ▶️ Run Locally

Clone repository:

```bash
git clone https://github.com/yourusername/Fraud-Detection-System.git
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

---

# 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
```

---

# 🔮 Future Improvements

* Add real-time transaction monitoring
* Add anomaly detection models
* Implement model explainability using SHAP
* Add Power BI fraud analytics dashboard

---

# 👨‍💻 Author

**Debdut Nandy**

---

⭐ If you found this project useful, consider giving it a star!
