# 🎬 Cinema Audience Forecasting using Machine Learning

## 📌 Overview
This project predicts daily **audience_count** for movie theatres using historical booking and visit data.

It is a **time-series regression problem** that combines:
- Customer visit data
- Theatre metadata
- Booking information
- Calendar-based features

The goal is to accurately forecast audience demand to help theatres optimize operations and planning.

---

## 🎯 Problem Statement
Predict the number of people (`audience_count`) attending a theatre on a given day using past data and contextual features.

---

## 📂 Dataset
Dataset sourced from a **Kaggle Competition – Cinema Audience Forecasting Challenge**

### Data includes:
- 📊 `booknow_visits` → daily audience count (target)
- 🎟️ `booking data` → tickets sold/booked
- 🏢 `theatre data` → type, location
- 📅 `date_info` → day of week

---

## 🛠️ Tech Stack
- Python 🐍
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- LightGBM, XGBoost

---

## 🔍 Project Workflow

### 1️⃣ Data Preprocessing
- Merged multiple datasets into a unified training dataset
- Converted date columns to datetime format
- Handled missing values:
  - Categorical → "Unknown"
  - Numerical → Median imputation

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Identified **right-skewed distribution** with strong outliers
- Observed **weekly seasonality (higher on weekends)**
- No strong monthly trend but high variability
- Detected **event-driven spikes in audience**

---

### 3️⃣ Feature Engineering

#### 📅 Date Features
- Day, Month, Year, Week, Quarter
- Weekend & Holiday indicators
- Friday release flag (movie release impact)
- Seasonal classification (Summer / Monsoon / Winter)

#### 🔄 Cyclical Encoding
- `day_sin`, `day_cos` for yearly patterns

#### 🏢 Theatre-Level Features
- Median audience per theatre (baseline)
- Total audience per theatre

#### ⏳ Time-Series Features
- Lag features: `lag_1, lag_7, lag_30, lag_90`
- Rolling means:
  - 3-day (short-term trend)
  - 7-day (weekly trend)
  - 30-day (monthly trend)
  - 90-day (long-term trend)

---

### 4️⃣ Data Splitting
- Time-based split (no leakage)
- Train: 80% (Jan 2023 – Dec 2023)
- Validation: 20% (Dec 2023 – Feb 2024)

---

### 5️⃣ Model Building

Models used:
- Linear Regression (Baseline)
- Random Forest
- Gradient Boosting
- LightGBM
- XGBoost

---

## 📊 Model Performance (R² Score)

| Model              | R² Score |
|--------------------|---------|
| Linear Regression  | 0.493 |
| Random Forest      | 0.492 |
| Gradient Boosting  | 0.486 |
| LightGBM           | 0.487 |
| XGBoost            | 0.479 |

---

## ⚙️ Hyperparameter Tuning
- Used **RandomizedSearchCV**
- Tuned LightGBM model

✅ **Best Validation Score: 0.502**

---

## 🚀 Final Model
- Model: **LightGBM (Tuned)**
- Achieved improved generalization with optimized parameters

---

## 📈 Key Insights
- Strong **weekly pattern** (weekends → high audience)
- High **variance & outliers** in data
- Theatre-specific patterns significantly affect predictions
- Lag & rolling features greatly improved performance

---

## 📁 Project Structure
