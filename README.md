# 📈 Credit Risk Assessment using Random Forest

This project focuses on building a **machine learning model** to predict the likelihood of a customer defaulting on a loan, based on demographic and financial information. We use the **Random Forest Classifier** algorithm, perform **data preprocessing**, **feature engineering**, and evaluate the model using performance metrics and visualizations.

## 📌 Project Overview

- **Goal**: Predict whether a loan applicant will default (binary classification).
- **Dataset**: [Credit Risk Dataset](https://www.kaggle.com/datasets)
- **ML Technique**: Random Forest Classifier
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Precision, Recall, F1-Score

## 🧰 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 🛠️ Project Workflow

### 1. **Data Loading & Preprocessing**
- Loaded the credit risk dataset.
- Dropped unnecessary unique identifier (`Id`).
- Performed one-hot encoding on categorical features: `Home`, `Intent`, and `Default`.
- Handled missing values using `SimpleImputer` with the mean strategy.

### 2. **Feature Selection**
- Selected relevant numerical and encoded features for modeling.

### 3. **Model Building**
- Split the dataset into training (80%) and testing (20%) sets.
- Trained a **Random Forest Classifier** with 100 trees (`n_estimators=100`) and default settings.

### 4. **Model Evaluation**
- Measured performance using:
  - **Accuracy Score**
  - **Classification Report** (Precision, Recall, F1-Score)
  - **Confusion Matrix**

### 5. **Visualization**
- Plotted **Feature Importance** to identify the most influential factors.
- Created a **Confusion Matrix** heatmap for better error analysis.

## 📈 Final Results

- **Accuracy Achieved**: ~91.74%
- **Precision for Non-Defaulters**: 0.91
- **Recall for Non-Defaulters**: 0.99
- **Precision for Defaulters**: 0.93
- **Recall for Defaulters**: 0.68
- **Overall Weighted F1-Score**: 0.91

*(The model predicts non-defaults with high accuracy but could be further optimized to improve recall for defaulters.)*

## 🚀 Future Improvements

- Tune hyperparameters using Grid Search or Randomized Search.
- Apply cross-validation for more robust validation.
- Handle class imbalance using SMOTE or adjusting class weights.
- Apply SHAP analysis for deeper model interpretability.
- Try alternative models like LightGBM, XGBoost, or CatBoost.
