# 🌾 Paddy Yield Predictor

## Project Overview

The **Paddy Yield Predictor** is a Machine Learning-based project that predicts the expected **paddy yield in kilograms** based on agricultural, soil, fertilizer, rainfall, temperature, wind, humidity, and other crop-related factors.

The project follows a complete Machine Learning lifecycle, including:

* Data loading
* Data understanding
* Exploratory Data Analysis (EDA)
* Data cleaning
* Data preprocessing
* Train-test splitting
* Model training
* Model comparison
* Hyperparameter tuning
* Model evaluation
* Model testing
* Prediction
* Deployment preparation

The final trained Machine Learning model can be used to predict paddy yield for new agricultural input data.

---

# Problem Statement

Paddy yield depends on several agricultural and environmental factors such as land area, paddy variety, soil type, fertilizer usage, rainfall, temperature, wind speed, humidity, and pest management.

It can be difficult to estimate crop yield accurately by considering all these factors manually.

This project aims to develop a **Machine Learning-based Paddy Yield Prediction System** that analyzes historical agricultural data and predicts the expected paddy yield.

The system can help in understanding the relationship between agricultural conditions and crop production and can support data-driven agricultural decision-making.

---

# Objectives

The main objectives of this project are:

* Load and understand the paddy yield dataset.
* Analyze the structure and quality of the dataset.
* Perform Exploratory Data Analysis (EDA).
* Identify missing values and duplicate records.
* Understand numerical and categorical features.
* Preprocess the agricultural data.
* Split the dataset into training and testing datasets.
* Train multiple Machine Learning regression models.
* Compare the performance of different models.
* Perform hyperparameter tuning.
* Evaluate the final model using regression metrics.
* Test the model with new input data.
* Save the trained model for future predictions.
* Develop a prediction application using Streamlit.

---

# Features

The Paddy Yield Predictor project includes:

* Automated dataset loading
* Data cleaning
* Missing-value analysis
* Duplicate-value checking
* Exploratory Data Analysis
* Numerical feature analysis
* Categorical feature analysis
* Data preprocessing pipeline
* Feature encoding
* Feature scaling where required
* Train-test split
* Multiple model training
* Model comparison
* Hyperparameter tuning
* Model evaluation
* Model testing
* Saved trained model
* Prediction functionality
* Logging implementation
* Utility functions
* Streamlit application support

---

# Machine Learning Approach

## Problem Type

This project is a **Regression Problem**.

The objective is to predict a continuous numerical value:

```text
Paddy yield(in Kg)
```

The model predicts the expected paddy production in kilograms based on the input agricultural features.

---

# Target Variable

| Feature              | Description                                  |
| -------------------- | -------------------------------------------- |
| `Paddy yield(in Kg)` | Paddy production/yield measured in kilograms |

---

# Dataset Information

The dataset contains **2,789 records** and agricultural/environmental features related to paddy cultivation.

The dataset contains information related to:

* Land area
* Agricultural block
* Paddy variety
* Soil type
* Seed rate
* Nursery information
* Fertilizer application
* Weed management
* Pest management
* Rainfall
* Irrigation-related measurements
* Temperature
* Wind speed
* Wind direction
* Relative humidity
* Crop-related inputs

---

# Dataset Features

| Feature                              | Description                                                      |
| ------------------------------------ | ---------------------------------------------------------------- |
| `Hectares`                           | Area of land used for paddy cultivation in hectares              |
| `Agriblock`                          | Agricultural block or area where cultivation takes place         |
| `Variety`                            | Paddy variety cultivated                                         |
| `Soil Types`                         | Type of soil present in the field                                |
| `Seedrate(in Kg)`                    | Quantity of seed used in kilograms                               |
| `LP_Mainfield(in Tonnes)`            | Main-field land preparation input                                |
| `Nursery`                            | Nursery-related information                                      |
| `Nursery area (Cents)`               | Nursery area measured in cents                                   |
| `LP_nurseryarea(in Tonnes)`          | Nursery land preparation input                                   |
| `DAP_20days`                         | DAP fertilizer application around 20 days                        |
| `Weed28D_thiobencarb`                | Weed-control treatment around 28 days                            |
| `Urea_40Days`                        | Urea fertilizer application around 40 days                       |
| `Potassh_50Days`                     | Potassium fertilizer application around 50 days                  |
| `Micronutrients_70Days`              | Micronutrient application around 70 days                         |
| `Pest_60Day(in ml)`                  | Pest-control treatment around 60 days                            |
| `30DRain( in mm)`                    | Rainfall measurement during the first crop period                |
| `30DAI(in mm)`                       | Agricultural/irrigation measurement during the first crop period |
| `30_50DRain( in mm)`                 | Rainfall measurement from day 30 to day 50                       |
| `30_50DAI(in mm)`                    | Agricultural/irrigation measurement from day 30 to day 50        |
| `51_70DRain(in mm)`                  | Rainfall measurement from day 51 to day 70                       |
| `51_70AI(in mm)`                     | Agricultural/irrigation measurement from day 51 to day 70        |
| `71_105DRain(in mm)`                 | Rainfall measurement from day 71 to day 105                      |
| `71_105DAI(in mm)`                   | Agricultural/irrigation measurement from day 71 to day 105       |
| `Min temp_D1_D30`                    | Minimum temperature during days 1–30                             |
| `Max temp_D1_D30`                    | Maximum temperature during days 1–30                             |
| `Min temp_D31_D60`                   | Minimum temperature during days 31–60                            |
| `Max temp_D31_D60`                   | Maximum temperature during days 31–60                            |
| `Min temp_D61_D90`                   | Minimum temperature during days 61–90                            |
| `Max temp_D61_D90`                   | Maximum temperature during days 61–90                            |
| `Min temp_D91_D120`                  | Minimum temperature during days 91–120                           |
| `Max temp_D91_D120`                  | Maximum temperature during days 91–120                           |
| `Inst Wind Speed_D1_D30(in Knots)`   | Wind speed during days 1–30                                      |
| `Inst Wind Speed_D31_D60(in Knots)`  | Wind speed during days 31–60                                     |
| `Inst Wind Speed_D61_D90(in Knots)`  | Wind speed during days 61–90                                     |
| `Inst Wind Speed_D91_D120(in Knots)` | Wind speed during days 91–120                                    |
| `Wind Direction_D1_D30`              | Wind direction during days 1–30                                  |
| `Wind Direction_D31_D60`             | Wind direction during days 31–60                                 |
| `Wind Direction_D61_D90`             | Wind direction during days 61–90                                 |
| `Wind Direction_D91_D120`            | Wind direction during days 91–120                                |
| `Relative Humidity_D1_D30`           | Relative humidity during days 1–30                               |
| `Relative Humidity_D31_D60`          | Relative humidity during days 31–60                              |
| `Relative Humidity_D61_D90`          | Relative humidity during days 61–90                              |
| `Relative Humidity_D91_D120`         | Relative humidity during days 91–120                             |
| `Trash(in bundles)`                  | Amount of agricultural trash measured in bundles                 |
| `Paddy yield(in Kg)`                 | Target variable representing paddy yield                         |

---

# Data Types

The dataset contains both numerical and categorical variables.

### Numerical Features

Numerical columns contain `int64` and `float64` values.

Examples:

```text
Hectares
Seedrate(in Kg)
LP_Mainfield(in Tonnes)
Nursery area (Cents)
DAP_20days
Urea_40Days
30DRain( in mm)
Min temp_D1_D30
Max temp_D1_D30
Relative Humidity_D1_D30
Trash(in bundles)
```

### Categorical Features

Categorical columns contain `object` values.

Examples:

```text
Agriblock
Variety
Soil Types
Nursery
Wind Direction_D1_D30
Wind Direction_D31_D60
Wind Direction_D61_D90
Wind Direction_D91_D120
```

---

# Technology Stack

## Programming Language

* Python

## Data Analysis

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn

## Model Saving

* Joblib / Pickle

## Application Framework

* Streamlit

## Environment

* Python Virtual Environment

---

# Project Structure

```text
Paddy_Yield_Predictor/
│
├── .venv/
│   └── Virtual environment
│
├── logs/
│   └── Application and training logs
│
├── models/
│   └── paddy_yield_predictor.pkl
│
├── notebooks/
│   │
│   ├── 01_Data_Loading.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Model_Comparison.ipynb
│   ├── 06_Hyperparameters.ipynb
│   ├── 07_Model_Evaluation.ipynb
│   └── 08_Model_Testing.ipynb
│
├── src/
│   │
│   ├── __pycache__/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model_utils.py
│   ├── prediction.py
│   └── training.py
│
├── paddydataset.csv
├── requirements.txt
├── app.py
└── README.md
```

---

# Notebook Description

## 01_Data_Loading.ipynb

This notebook performs the initial dataset loading and understanding.

Performed operations include:

* Importing required libraries
* Loading the CSV dataset
* Checking dataset shape
* Viewing the first and last records
* Checking column names
* Checking data types
* Checking missing values
* Checking duplicate records
* Understanding numerical and categorical columns

---

# 02_EDA.ipynb

This notebook performs Exploratory Data Analysis.

The analysis includes:

* Descriptive statistics
* Missing-value analysis
* Duplicate analysis
* Target variable analysis
* Distribution of numerical features
* Histogram analysis
* Boxplot analysis
* Outlier analysis
* Categorical feature analysis
* Correlation analysis
* Relationship between features and paddy yield

EDA helps identify important patterns and relationships in the agricultural dataset before model training.

---

# 03_Preprocessing.ipynb

This notebook prepares the dataset for Machine Learning.

Performed operations include:

* Data cleaning
* Removing unnecessary columns
* Handling missing values
* Separating features and target
* Identifying numerical features
* Identifying categorical features
* Encoding categorical variables
* Scaling numerical features where required
* Building preprocessing pipelines

The target variable is:

```text
Paddy yield(in Kg)
```

---

# 04_Model_Training.ipynb

This notebook focuses on training Machine Learning regression models.

Performed operations include:

* Separating X and y
* Train-test split
* Building preprocessing pipeline
* Training regression models
* Generating predictions
* Saving trained models

The project primarily uses **Random Forest Regression** for paddy yield prediction.

---

# 05_Model_Comparison.ipynb

This notebook compares different regression algorithms.

The models can be evaluated based on:

* MAE
* MSE
* RMSE
* R² Score

The best-performing model is selected based on the evaluation results.

---

# 06_Hyperparameters.ipynb

This notebook performs hyperparameter optimization.

Hyperparameter tuning is used to find better model configurations and improve prediction performance.

Parameters can include:

* Number of estimators
* Maximum depth
* Minimum samples split
* Minimum samples leaf
* Maximum features

Techniques such as:

* Grid Search
* Randomized Search

can be used for optimization.

---

# 07_Model_Evaluation.ipynb

This notebook evaluates the trained regression model.

The main evaluation metrics are:

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error — MSE

Measures the average squared difference between actual and predicted values.

### Root Mean Squared Error — RMSE

The square root of MSE and represents prediction error in the same unit as the target.

### R² Score

Measures how well the model explains the variation in the target variable.

---

# 08_Model_Testing.ipynb

This notebook tests the final trained model using new agricultural input data.

The process includes:

* Loading the saved model
* Creating sample input data
* Applying the preprocessing pipeline
* Generating predictions
* Displaying predicted paddy yield

The output is represented in:

```text
Kilograms (Kg)
```

---

# Source Code Description

## `src/data_loader.py`

Responsible for:

* Loading the dataset
* Reading CSV files
* Basic dataset handling
* Data validation

---

## `src/training.py`

Responsible for:

* Preparing training data
* Splitting the dataset
* Building the Machine Learning pipeline
* Training the model
* Saving the trained model

---

## `src/model_utils.py`

Contains reusable Machine Learning utility functions used during model development and training.

---

## `src/prediction.py`

Responsible for:

* Loading the saved model
* Preparing input data
* Generating predictions
* Returning predicted paddy yield

---

## `src/__init__.py`

Initializes the `src` package and allows the modules inside `src` to be imported.

---

# Machine Learning Pipeline

```text
                Paddy Dataset
                      |
                      ↓
                Data Loading
                      |
                      ↓
              Data Understanding
                      |
                      ↓
                     EDA
                      |
                      ↓
              Data Cleaning
                      |
                      ↓
              Data Preprocessing
                      |
                      ↓
             Feature Engineering
                      |
                      ↓
                Train-Test Split
                      |
                      ↓
               Model Training
                      |
                      ↓
              Model Comparison
                      |
                      ↓
             Hyperparameter Tuning
                      |
                      ↓
              Model Evaluation
                      |
                      ↓
                Model Testing
                      |
                      ↓
             Save Trained Model
                      |
                      ↓
                Prediction
```

---

# Model Evaluation Metrics

Since this is a **regression problem**, the following metrics are used:

| Metric   | Purpose                                               |
| -------- | ----------------------------------------------------- |
| MAE      | Measures average absolute prediction error            |
| MSE      | Measures average squared prediction error             |
| RMSE     | Measures prediction error in target units             |
| R² Score | Measures how well the model explains target variation |

---

# Logging

The project contains a `logs` directory for storing application and Machine Learning process logs.

Logging can help track:

* Dataset loading
* Preprocessing steps
* Model training
* Model saving
* Prediction operations
* Errors and exceptions

---

# Model Saving

The trained model is saved inside the `models` directory.

```text
models/
└── paddy_yield_predictor.pkl
```

The saved model can be loaded later without retraining the entire Machine Learning model.

---

# Streamlit Application

The project includes a Streamlit application that can be used to provide agricultural information and generate paddy yield predictions.

The application allows users to enter values for the features used during model training.

The application then:

1. Collects user input.
2. Creates a DataFrame.
3. Arranges the features in the correct order.
4. Sends the input through the trained Machine Learning pipeline.
5. Generates the predicted paddy yield.
6. Displays the predicted yield in kilograms.

Example output:

```text
Predicted Paddy Yield

XXXX.XX Kg
```

---

# Installation and Setup

## 1. Clone the Repository

```bash
git clone <repository-url>
```

---

## 2. Navigate to the Project

```bash
cd Paddy_Yield_Predictor
```

---

## 3. Create Virtual Environment

```bash
python -m venv .venv
```

---

## 4. Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project

If the model has not already been trained, run the training process first.

```bash
python src/training.py
```

After successful training, the trained model should be available inside:

```text
models/
```

Then start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

---

# Requirements

The main Python libraries used in this project include:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

The complete dependencies are available in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# Future Enhancements

Future improvements may include:

* Deploying the application to Streamlit Cloud.
* Adding more agricultural datasets.
* Improving model accuracy.
* Adding feature-importance visualization.
* Adding prediction history.
* Adding downloadable prediction reports.
* Adding interactive agricultural dashboards.
* Providing recommendations based on predicted yield.
* Adding weather API integration.
* Adding real-time agricultural information.
* Comparing additional Machine Learning algorithms.

---

# Conclusion

The **Paddy Yield Predictor** demonstrates how Machine Learning can be applied to agricultural data to predict paddy production.

The project covers the complete Machine Learning workflow, starting from **data loading and exploratory data analysis** through **preprocessing, model training, model comparison, hyperparameter tuning, evaluation, testing, and prediction**.

The system uses agricultural and environmental factors such as land area, paddy variety, soil type, fertilizer application, rainfall, temperature, wind, and humidity to estimate paddy yield.

This project demonstrates the practical application of **Python, Pandas, NumPy, Scikit-learn, Machine Learning, and Streamlit** in an agriculture-focused prediction system.

---

# Author

**Truptimayee Khuntia**

**Paddy Yield Predictor**

Machine Learning Project

---

# License

This project is created for educational and Machine Learning project purposes.
