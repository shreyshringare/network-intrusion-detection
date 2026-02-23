# Network Intrusion Detection System using Machine Learning

## Overview

This project implements a Machine Learning based Network Intrusion Detection System using the NSL-KDD dataset. The system analyzes network traffic data and classifies it as either normal traffic or an intrusion. It also supports multiclass classification to identify specific types of attacks.

The project demonstrates the complete machine learning workflow including data preprocessing, model training, evaluation, and deployment using Streamlit.

---

## Features

- Data preprocessing and cleaning
- Feature encoding and scaling
- Feature importance analysis
- Binary classification (Normal vs Intrusion)
- Multiclass classification (Attack type detection)
- Model comparison and evaluation
- Hyperparameter tuning
- Cross-validation
- Web application using Streamlit
- Real-time prediction simulation

---

## Machine Learning Models Used

The following models were implemented and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier (Final model)
- Gradient Boosting Classifier
- Support Vector Machine (SVM)

Random Forest was selected as the final model due to its high accuracy and robustness.

---

## Dataset

Dataset used: NSL-KDD Dataset

The dataset contains network traffic features such as:

- duration
- protocol_type
- service
- flag
- src_bytes
- dst_bytes
- count
- srv_count
- and other network traffic features

Output labels include normal traffic and multiple attack types.

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook

---

network-intrusion-detection/

app.py
intrusion_detection.ipynb
predict.py
requirements.txt
README.md
.gitignore
model/ (contains trained model files, not uploaded to GitHub)


---

## How to Run the Project

### Step 1: Clone the repository


git clone https://github.com/shreyshringare/network-intrusion-detection.git

cd network-intrusion-detection


### Step 2: Install dependencies


pip install -r requirements.txt


### Step 3: Run the Streamlit application


streamlit run app.py


### Step 4: Open in browser


http://localhost:8501


---

## Example Output

Input network traffic features are provided through the web interface.

The system predicts:

- Normal traffic  
or  
- Intrusion attack type  

along with prediction confidence.

---
