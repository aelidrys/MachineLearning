# 🚕 NYC Taxi Trip Duration Prediction *(Ridge Regression & Geospatial Engineering)*
---
### [View my complete solution on Kaggle](https://www.kaggle.com/code/ayoubedark78/tripdurationprediction)
[⬅️ Back to Main ML Portfolio](../../README.md)

<p align="center">
  <img src="https://storage.googleapis.com/kaggle-media/competitions/kaggle/3333/media/taxi_meter.png" width="400" alt="Taxi Meter">
</p>

## 📋 Overview
- **The Problem:** Predicting the total ride duration of taxi trips in New York City using data released by the NYC Taxi and Limousine Commission.
- **The Challenge:** Raw data contains timestamps and geospatial coordinates, which cannot be fed directly into a regression model. The project requires heavy datetime parsing and geospatial feature engineering to extract meaningful patterns (e.g., rush hour traffic, trip distances).

### 🗃️ Data Dictionary
| Feature | Description |
| :--- | :--- |
| `id` | Unique identifier for each trip |
| `vendor_id` | Code indicating the provider associated with the trip record |
| `pickup_datetime` | Date and time when the meter was engaged |
| `dropoff_datetime` | Date and time when the meter was disengaged *(Target derived from this)* |
| `passenger_count` | Number of passengers in the vehicle |
| `pickup/dropoff_longitude` | Geo-coordinates where the meter was engaged/disengaged |
| `pickup/dropoff_latitude` | Geo-coordinates where the meter was engaged/disengaged |
| `store_and_fwd_flag` | Indicates if the record was held in vehicle memory before sending (`Y`/`N`) |

---

## 🛠️ Workflow & Methodology

### 🧹 1. Data Cleaning
* **Quality Assurance:** Assessed the dataset for missing values, extreme outliers (e.g., trips lasting multiple days or 0 seconds), and invalid categorical data.
* **Outlier Handling:** Filtered out anomalies to ensure the regression model trains on realistic, representative NYC traffic patterns.

### 🧬 2. Feature Engineering
* **Geospatial Distance Calculation:** Engineered a new `distance` feature derived from the raw latitude and longitude coordinates, providing the model with a direct spatial metric instead of abstract coordinate points.
* **Temporal Feature Extraction:** Parsed the `pickup_datetime` timestamp into highly granular temporal features: `pickup_hour`, `pickup_month`, and `pickup_weekday`. This allows the model to learn cyclical traffic patterns (e.g., Friday rush hour vs. Sunday morning).

### 🧠 3. Model Training
* **Feature Scaling:** Applied Scikit-Learn's `MinMaxScaler` to normalize the engineered features, ensuring the distance and temporal features exist on a uniform scale.
* **Polynomial Expansion:** Generated interaction terms using `PolynomialFeatures` to allow the linear algorithm to capture non-linear relationships.
* **Algorithm:** Fitted a **Ridge Regression** model (L2 Regularization) to handle the multicollinearity introduced by the polynomial features and prevent overfitting.

---

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **Training R-Squared ($R^2$)** | **68.81%** |
| **Validation R-Squared ($R^2$)** | **68.81%** |

> **Takeaway:** The identical training and validation scores indicate a perfectly generalized model with zero overfitting. The model successfully extracts the baseline linear patterns from the engineered temporal and geospatial features.