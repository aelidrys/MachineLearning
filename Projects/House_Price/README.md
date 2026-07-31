# 🏠 House Price Prediction *(Linear Regression & Feature Engineering)*
---
### [View my complete solution on Kaggle](https://www.kaggle.com/code/ayoubedark78/housepriceprediction)
[⬅️ Back to Main ML Portfolio](../../README.md)

<p align="center">
  <img src="https://www.propertyreporter.co.uk/images/660x350/16402-shutterstock_538341163.jpg" width="600" alt="House Price Prediction">
</p>

## 📋 Overview
- **The Problem:** Accurately predicting the sale price of a home is a classic continuous regression problem. It requires careful handling of dozens of variables describing every aspect of residential homes.
- **The Goal:** Build a highly accurate predictive model by heavily focusing on data cleaning, imputation, and feature engineering to extract the maximum amount of signal from the provided dataset.

---

## 🛠️ Workflow & Methodology

### 🧹 1. Data Preprocessing & Cleaning
* **Missing Value Treatment:** 
  * Automatically dropped columns with a missing data rate of **> 30%** to prevent introducing severe bias.
  * Imputed remaining missing values using statistical strategies (Mean, Median, Mode) or assigning logical defaults (e.g., "None Available" for missing categorical features like Pool or Fence).
* **Outlier Treatment:** Applied the **Interquartile Range (IQR)** algorithm to detect and cap extreme outliers, preventing them from skewing the regression weights.
* **Categorical Encoding:** Utilized Scikit-Learn's `LabelEncoder` to translate categorical strings into numerical formats readable by the machine learning algorithm.

### 🧬 2. Feature Selection & Engineering
* **Correlation Filtering:** Selected the final feature set based on strict Pearson correlation scores against the target variable (`SalePrice`).
* **Polynomial Feature Generation:** Applied Scikit-Learn's `PolynomialFeatures` to generate interaction terms and non-linear features. This is a critical step that allows a standard linear model to capture complex, non-linear relationships in the housing data.

### 🧠 3. Model Training & Evaluation
* **Data Splitting & Scaling:** Split the dataset into Train and Test subsets and applied Feature Scaling to ensure all variables contributed equally to the model penalty.
* **Algorithm:** Trained a robust Linear Regression model using the engineered polynomial features.

---

## 📊 Model Performance

By focusing heavily on the data preprocessing and feature engineering phases, the linear model achieved excellent predictive variance:

| Metric | Score |
| :--- | :--- |
| **Training R-Squared ($R^2$)** | **92.03%** |
| **Testing R-Squared ($R^2$)** | **89.81%** |

> **Takeaway:** The tight grouping between the training score (92%) and testing score (89.8%) indicates that the model is well-generalized and avoids overfitting, despite the added complexity of polynomial features.