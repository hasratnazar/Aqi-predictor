# 🌍 AQI Predictor

An end-to-end **Air Quality Index (AQI) Prediction System** that fetches live weather and pollution data, processes it through a machine learning pipeline, and predicts air quality levels using models such as **Gradient Boosting** and **Random Forest**.

The project integrates with the **Hopsworks Feature Store** for seamless data versioning, model registry, and metric tracking.  
The system is automated using **GitHub Actions**, enabling continuous data fetching, preprocessing, training, and deployment.


## ✨ Key Features

 **Automated Data Collection** – Fetches real-time weather and pollution data using APIs.  
 **Feature Store Integration** – Uses **Hopsworks** for feature versioning, storage, and tracking.  
 **Machine Learning Models** – Implements **Random Forest** and **Gradient Boosting** for AQI prediction.  
 **Streamlit Dashboard** – Provides an interactive UI for real-time AQI prediction.  
 **CI/CD Pipeline** – GitHub Actions automate data updates, model training, and deployment.  
 **Scalable and Modular** – Clean, modular pipeline scripts for fetching, preprocessing, and training.  
 **End-to-End MLOps Setup** – Continuous data ingestion, training, evaluation, and monitoring.  



## 🧠 Project Overview

The **AQI Predictor** performs the following tasks:

1. **Data Fetching** – Collects real-time weather and pollution data (using OpenWeather API).  
2. **Feature Engineering** – Processes and prepares data for model training.  
3. **Model Training** – Trains Gradient Boosting and Random Forest models, stores metrics and artifacts in Hopsworks.
4. **Prediction App** – Streamlit-based dashboard that predicts AQI levels from input weather parameters.  
5. **Automation** – GitHub Actions workflows handle periodic data fetching, preprocessing, and retraining.



## 🧰 Tech Stack

| Component  | Technology Used |
|------------|----------------|
| **Programming Language** | Python 3.x |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (RandomForest, GradientBoosting) |
| **Feature Store** | Hopsworks |
| **Web Framework** | Streamlit |
| **Automation** | GitHub Actions |
| **Environment Management** | Virtualenv |
| **Data Source APIs** | OpenWeatherMap / Air Pollution API |

**Languages & Libraries**
- 🐍 Python (Pandas, NumPy, Scikit-learn, Matplotlib)
- 📊 Streamlit – Interactive web app
- ☁️ Hopsworks – Feature Store and Model Registry
- 🌤 OpenWeather API – Weather and pollution data source
- 🧪 Joblib – Model serialization
- 🔄 GitHub Actions – Continuous Integration & Deployment


## ⚙️ How the Pipeline Works

The project follows a modular, end-to-end machine learning pipeline located inside the `/pipeline` directory.

### 🧩 1. 'fetch_raw.py'
- Fetches 30 days of weather and pollutant data from the OpenWeatherMap API. 

### 🧹 2. 'ml_feature.py'
- Cleans, merges, and engineers features from raw data.
- Stores processed dataset locally and uploads to **Hopsworks Feature Store**.

### 🧠 3. 'training.py'
- Loads cleaned data from the Feature Store.
- Trains multiple machine learning models (Random Forest, Gradient Boosting).

### 🌐 4. 'app.py'
- Streamlit web app for real-time AQI prediction.
- Fetches live data 
- Displays AQI prediction, pollutant breakdown, and data visualization.


## 🧑‍💻 How to Run the Project Locally

### 1 Clone the Repository
```bash
git clone https://github.com/<your-username>/aqi-predictor.git
cd "AQI PREDICTOR"

### 2 create and activate virtual environment

python -m venv venv
venv\Scripts\activate

### 3 Install Dependencies

pip install -r requirements.txt

### 4 Run the Pipeline

### 5 Launch the Streamlit App

streamlit run app.py





