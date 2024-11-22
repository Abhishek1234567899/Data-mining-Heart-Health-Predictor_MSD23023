# Heart Disease Prediction

**Website**: [Heart Disease Prediction App](https://heart-health-predictor-g1k5.onrender.com)

This project is a machine learning-based application designed to predict the likelihood of heart disease in a patient based on various health attributes. The primary goal is to assist healthcare professionals by providing data-driven insights for early diagnosis and prevention.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Model](#model)  
5. [Hyperparameter Tuning](#hyperparameter-tuning)  
6. [Installation](#installation)  
7. [Usage](#usage)  
8. [Evaluation Metrics](#evaluation-metrics)  
9. [Future Improvements](#future-improvements)  
10. [Contributing](#contributing)  
11. [License](#license)

---

## Overview

Cardiovascular diseases are among the leading causes of death worldwide. Early diagnosis is critical in reducing mortality rates. This project leverages machine learning to analyze patient data and predict the likelihood of heart disease, enabling early detection and intervention. The web-based application allows users to input health attributes and instantly receive predictions, offering an intuitive tool for both healthcare providers and researchers.

---

## Features

- **Multi-Model Support**: Implements Logistic Regression, Decision Trees, Random Forest, SVM, and KNN for comparison.  
- **Hyperparameter Optimization**: Models are fine-tuned for improved performance.  
- **Data Visualizations**: Insights are provided through graphs and charts for exploratory analysis.  
- **User-Friendly Web Application**: Easy-to-use interface for quick predictions.  
- **Open-Source**: The code is freely available for customization and extension.

---

## Dataset

The **Heart Disease UCI Dataset** serves as the foundation for this project. It contains 303 records with 14 medical attributes relevant to heart health.  

### Key Features in the Dataset:
- `age`: Age of the patient.  
- `sex`: Gender (1 = male, 0 = female).  
- `cp`: Chest pain type (0–3).  
- `trestbps`: Resting blood pressure (in mm Hg).  
- `chol`: Serum cholesterol level (in mg/dL).  
- `fbs`: Fasting blood sugar (>120 mg/dL, 1 = true, 0 = false).  
- `restecg`: Resting electrocardiographic results (0–2).  
- `thalach`: Maximum heart rate achieved.  
- `exang`: Exercise-induced angina (1 = yes, 0 = no).  
- `oldpeak`: ST depression induced by exercise relative to rest.  
- `slope`: Slope of the peak exercise ST segment.  
- `ca`: Number of major vessels (0–3) colored by fluoroscopy.  
- `thal`: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect).  

The dataset is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

---

## Model

This project implements and evaluates the following machine learning models:
- **Logistic Regression**  
- **Decision Trees**  
- **Random Forest**  
- **Support Vector Machines (SVM)**  
- **K-Nearest Neighbors (KNN)**  

### Workflow
1. Data Preprocessing: Handling missing values, encoding categorical variables, and scaling.  
2. Exploratory Data Analysis: Visualizations to understand feature relationships.  
3. Model Training: Training multiple classifiers with cross-validation.  
4. Model Evaluation: Metrics include accuracy, precision, recall, and F1-score.  
5. Deployment: The best-performing model is deployed in the web application.

---

## Hyperparameter Tuning

To enhance model accuracy and efficiency, hyperparameter tuning was performed using **GridSearchCV** and **RandomizedSearchCV**. Below are the optimized parameters:

### Logistic Regression
- `penalty`: `['l1', 'l2', 'elasticnet']`  
- `C`: `[0.01, 0.1, 1, 10, 100]`  
- `solver`: `['liblinear', 'saga']`  

### Decision Tree
- `criterion`: `['gini', 'entropy']`  
- `max_depth`: `[None, 5, 10, 15, 20]`  
- `min_samples_split`: `[2, 5, 10]`  
- `min_samples_leaf`: `[1, 2, 5]`  

### Random Forest
- `n_estimators`: `[50, 100, 200, 500]`  
- `max_depth`: `[None, 5, 10, 15, 20]`  
- `min_samples_split`: `[2, 5, 10]`  
- `min_samples_leaf`: `[1, 2, 4]`  

### Support Vector Machine (SVM)
- `C`: `[0.1, 1, 10, 100]`  
- `kernel`: `['linear', 'rbf', 'poly', 'sigmoid']`  
- `gamma`: `['scale', 'auto']`  

### K-Nearest Neighbors (KNN)
- `n_neighbors`: `[3, 5, 7, 9]`  
- `weights`: `['uniform', 'distance']`  
- `metric`: `['euclidean', 'manhattan']`  

---

## Installation

### Prerequisites
Ensure the following are installed on your system:
- Python 3.x  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `flask`

### Steps
1. Clone the repository:
   ```bash
  https://github.com/Abhishek1234567899/Data-mining-Heart-Health-Predictor_MSD23023.git


Install dependencies:
pip install -r requirements.txt

Run the application:

python app.py
