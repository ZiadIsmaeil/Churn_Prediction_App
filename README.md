# Customer Churn Intelligence

An end-to-end machine learning project for predicting customer churn and explaining individual predictions using Explainable AI (SHAP).

## 📌 Project Overview

Customer churn is a major business problem for subscription-based companies.

This project develops a complete machine learning workflow that can:

- Predict customer churn probability
- Identify customers who may be at higher risk of churn
- Analyze factors associated with churn
- Explain individual predictions using SHAP
- Provide an interactive Streamlit dashboard

## 💼 Business Problem

Customer churn is an imbalanced classification problem, where customers who stay typically outnumber customers who leave.

Because of this imbalance, accuracy alone is not sufficient for evaluating a churn model.

This project therefore focuses on:

- Precision
- Recall
- F1 Score
- ROC-AUC

The classification threshold was also evaluated to understand the trade-off between identifying more potential churners and generating false positives.

## 📊 Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer demographic information, services, contract details, payment methods, tenure, and billing information.

### Target Variable

Churn

No → 0
Yes → 1

### Main Features

- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Payment Method
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming Services
- Senior Citizen Status
- Partner
- Dependents

## 🧹 Data Cleaning

The following preprocessing steps were performed:

1. Converted TotalCharges to numeric values
2. Handled invalid and blank TotalCharges values
3. Removed the customerID identifier
4. Removed duplicate records
5. Separated features and target
6. Performed a stratified train/test split
7. Applied preprocessing after the train/test split to avoid data leakage

Final cleaned dataset:

7,021 rows
20 columns

## 🔎 Exploratory Data Analysis

Several customer characteristics showed meaningful differences between churned and non-churned customers.

### Contract Type

Month-to-month customers showed substantially higher churn than customers with one-year or two-year contracts.

### Internet Service

Fiber optic customers showed higher churn than DSL customers and customers without internet service.

### Payment Method

Electronic check customers showed a noticeably higher churn rate than the other payment methods.

### Tenure

Customers who churned had a lower average tenure than customers who stayed.

### Monthly Charges

Customers who churned had higher average monthly charges.

These relationships are descriptive associations from the dataset and should not be interpreted as causal effects.

## 🤖 Machine Learning

Several classification models were evaluated.

### Logistic Regression

Accuracy: 80.2%
Precision: 66.0%
Recall: 52.2%
F1 Score: 58.3%
ROC-AUC: 84.0%

### Random Forest

Accuracy: 77.9%
Precision: 61.3%
Recall: 44.4%
F1 Score: 51.5%
ROC-AUC: 81.9%

### XGBoost

Accuracy: 80.1%
Precision: 65.6%
Recall: 51.9%
F1 Score: 58.0%
ROC-AUC: 84.0%

## ⚖️ Handling Class Imbalance

Because churn represents a minority class, a balanced Logistic Regression model was evaluated using:

class_weight="balanced"

Final balanced model results:

Accuracy: 74.1%
Precision: 50.7%
Recall: 78.0%
F1 Score: 61.4%
ROC-AUC: 84.0%

The balanced model increased recall while reducing precision, demonstrating the precision-recall trade-off involved in imbalanced classification.

## 🎯 Threshold Analysis

The classification threshold was evaluated at several values.

Threshold 0.30:
Precision: 42.5%
Recall: 94.1%
F1 Score: 58.6%

Threshold 0.40:
Precision: 46.6%
Recall: 86.3%
F1 Score: 60.5%

Threshold 0.50:
Precision: 50.7%
Recall: 78.0%
F1 Score: 61.4%

Threshold 0.60:
Precision: 54.9%
Recall: 68.3%
F1 Score: 60.8%

Threshold 0.70:
Precision: 61.4%
Recall: 55.1%
F1 Score: 58.1%

For this project, a threshold of 0.50 was selected based on the evaluated F1 results.

In a production environment, the threshold should be tuned according to the business cost of false positives, false negatives, and available retention resources.

## 🔎 Explainable AI

The project uses SHAP (SHapley Additive exPlanations) to explain individual predictions.

SHAP shows how individual features shift the model output relative to the model's background/reference data.

Examples of features that can influence an individual prediction include:

- Tenure
- Contract Type
- Internet Service
- Payment Method
- Monthly Charges
- Technical Support
- Streaming Services

Positive SHAP contributions move the model output toward higher predicted churn.

Negative SHAP contributions move the model output toward lower predicted churn.

SHAP describes model behavior and should not be interpreted as proof of causal effects.

## 🧪 SHAP Validation

The SHAP explanation was technically validated to ensure that the feature contributions reconstruct the model output.

For the tested customer:

Model Output:
-0.774100

SHAP Base Value:
-0.549681

Sum of SHAP Contributions:
-0.224419

Reconstructed Output:
-0.774100

Difference:
0.0000000000

The reconstructed output matches the model output within numerical precision.

This confirms that the SHAP explanation is mathematically consistent with the model prediction.

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application for customer churn analysis.

The application allows users to:

- Enter customer information
- Generate a churn probability
- View the predicted classification
- Inspect the most influential features
- Explore SHAP explanations
- View a SHAP waterfall plot
- Review model performance
- Inspect model configuration

Application sections:

1. Prediction
2. Explainability
3. Model Performance

## 🏗️ Model Architecture

Raw Customer Data
        ↓
Data Cleaning
        ↓
Train / Test Split
        ↓
Feature Preprocessing
        ↓
Numerical Features → StandardScaler
        ↓
Categorical Features → OneHotEncoder
        ↓
Balanced Logistic Regression
        ↓
Churn Probability
        ↓
Decision Threshold
        ↓
Churn Prediction
        ↓
SHAP Explainability

## 🛠️ Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Explainable AI

- SHAP

### Application

- Streamlit

### Model Persistence

- Joblib

### Development

- Jupyter Notebook
- VS Code
- Git
- GitHub

## 📁 Project Structure

Churn_Prediction_App/
│
├── app/
│   └── app.py
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── threshold.json
│   └── X_train_background.pkl
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt

## 🚀 Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>

Navigate to the project:

cd Churn_Prediction_App

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## ▶️ Run the Application

From the project root:

streamlit run app/app.py

The Streamlit application will open in your browser.

## ⚠️ Model Limitations

This project is designed as a machine learning portfolio project and demonstrates an end-to-end Data Science workflow.

Important limitations include:

- The dataset represents a specific telecommunications customer population.
- Historical relationships do not necessarily imply causation.
- Model performance may change on new populations or future data.
- The selected threshold is not optimized for a specific company's retention economics.
- SHAP explains the model's behavior rather than proving why a customer will churn.

A production system would require additional validation, monitoring, data quality checks, retraining procedures, and business-specific threshold optimization.

## 🔮 Future Improvements

Potential improvements include:

- Cross-validation
- MLflow experiment tracking
- FastAPI model serving
- Docker deployment
- Cloud deployment
- Automated model retraining
- Model monitoring
- Customer segmentation
- Business-cost-based threshold optimization

## 🎯 Project Objective

This project demonstrates a complete Data Science and Machine Learning workflow:

Business Problem
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Preprocessing
        ↓
Model Development
        ↓
Model Comparison
        ↓
Class Imbalance Handling
        ↓
Threshold Analysis
        ↓
Explainable AI
        ↓
Model Validation
        ↓
Interactive Deployment

The objective is not only to build a predictive model, but to demonstrate how a machine learning solution can be developed, evaluated, explained, and delivered through an interactive application.

