# Tuberculosis Survival Prediction

This project is a machine learning-based application designed to predict the survival outcome of tuberculosis (TB) patients using clinical and demographic data. By analyzing patient attributes, the model can forecast whether an infected individual is likely to survive, helping healthcare professionals make more informed decisions.

---

## 🩺 Project Description

Tuberculosis (TB) is a potentially deadly infectious disease, especially in regions with limited healthcare resources. While early detection and treatment can improve recovery rates, knowing the **survival probability** can further enhance patient care strategies.

This project applies supervised machine learning algorithms to predict patient survival outcomes based on medical records, symptoms, and test results. The goal is to assist doctors and healthcare workers in prioritizing care and improving survival rates through early risk assessment.

---

## 📂 Dataset

The dataset is stored in CSV format and contains:

- **Features**  
  - Age  
  - Gender  
  - Symptoms (e.g., cough, chest pain, fever)  
  - Test results (e.g., X-ray findings, sputum test)  
  - Medical history  

- **Target Variable**  
  - `1` — Patient survived TB  
  - `0` — Patient did not survive

---

## 💻 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/stanlee47/Tuberculosis
    cd tuberculosis
    ```

2. Install required dependencies:
    ```bash
    pip install -r req.txt
    ```

---

## ⚙️ Usage

1. Prepare your dataset and place it in the project directory as `data.csv`.
2. Train the model:
    ```bash
    python src/train.py
    ```

The model will output whether the patient is likely to survive or not, along with the probability.

---

## 🧠 Model

The project includes several machine learning algorithms for performance comparison:

- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machine (SVM)  

Model selection is based on metrics like Accuracy, Precision, Recall, and F1-Score.


---

## 📊 Results

The trained models have shown strong predictive power, with the best-performing model achieving high accuracy in predicting survival outcomes on unseen data.  

Evaluation reports and confusion matrices are available in the `model/` directory.
and experiment tracking in
    ```
        https://dagshub.com/stanlykurian22/Tuberculosis/experiments
    ```
