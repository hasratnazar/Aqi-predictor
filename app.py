import streamlit as st
import pandas as pd
import os
import requests
import datetime
import joblib
import numpy as np
from dotenv import load_dotenv

# Function to Load LOCAL Models
@st.cache_resource
def load_local_models():
    """
    Loads models and scalers from the local 'model/' directory.
    """
    with st.spinner("Loading local models..."):
        models = {}
        scalers = {}
        model_names = ["gradient_boosting", "RandomForest"]  
        base_path = "model"  

        try:
            for name in model_names:
                model_path = os.path.join(base_path, name, "model.pkl")
                scaler_path = os.path.join(base_path, name, "scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    models[name] = joblib.load(model_path)
                    scalers[name] = joblib.load(scaler_path)
                else:
                    st.error(f"Missing files for '{name}' in '{os.path.join(base_path, name)}/' directory.")
            
            if not models:
                st.error("No models were loaded. Check your 'model' directory structure.")
                return None, None

            st.success("Local models loaded successfully.")
            return models, scalers
        except Exception as e:
            st.error(f"Fatal error loading local models: {e}")
            return None, None


# Function to Get Forecast Data for Specific Hour 
@st.cache_data(ttl=600)
def get_single_forecast_data(hours_ahead, owm_api_key):
    """
    Fetches pollutant + weather data for a specific hour in the future.
    """
    LAT = 24.8607
    LON = 67.0011

    try:
        
        pollution_url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution/forecast"
            f"?lat={LAT}&lon={LON}&appid={owm_api_key}"
        )
        response_poll = requests.get(pollution_url)
        response_poll.raise_for_status()
        poll_data = response_poll.json()['list']

        
        poll_forecast = poll_data[hours_ahead - 1]
        poll_dt = poll_forecast['dt']
        forecast_time = datetime.datetime.fromtimestamp(poll_dt, tz=datetime.timezone.utc)

        
        weather_url = (
            f"http://api.openweathermap.org/data/2.5/forecast"
            f"?lat={LAT}&lon={LON}&appid={owm_api_key}&units=metric"
        )
        response_weather = requests.get(weather_url)
        response_weather.raise_for_status()
        weather_data_list = response_weather.json()['list']

        
        closest_weather = min(weather_data_list, key=lambda x: abs(x['dt'] - poll_dt))

        feature_row = [
            poll_forecast['components']['pm2_5'],
            poll_forecast['components']['pm10'],
            poll_forecast['components']['o3'],
            closest_weather['main']['temp'],
            forecast_time.hour,
            forecast_time.day
        ]

        display_data = {
            "Forecast Time": forecast_time.strftime('%Y-%m-%d %H:%M'),
            "pm2_5": poll_forecast['components']['pm2_5'],
            "pm10": poll_forecast['components']['pm10'],
            "o3": poll_forecast['components']['o3'],
            "temp": closest_weather['main']['temp'],
            "hour": forecast_time.hour,
            "day": forecast_time.day
        }

        return feature_row, display_data

    except Exception as e:
        st.error(f"Error fetching data from OpenWeather: {e}")
        return None, None


# Streamlit UI Configuration 
st.set_page_config(
    page_title="AQI Forecast",
    page_icon="🌤️",
    layout="centered"
)

FEATURE_NAMES = ['pm2_5', 'pm10', 'o3', 'temp', 'hour_of_day', 'day_of_month']

# Load API Key
try:
    load_dotenv()
    OWM_API_KEY = os.environ["OPENWEATHER_API_KEY"]
except KeyError:
    st.error("API_KEY not found in .env file. Please add it.")
    OWM_API_KEY = None

# Load Models
models, scalers = load_local_models()

#Main App
st.title("AQI Forecast Tool 🌤️")
st.info("Predicts Air Quality Index (AQI) for the next 3 days using pollutant and weather data.")

if not OWM_API_KEY or not models or not scalers:
    st.error("App is not configured. Check your API key or model folder structure.")
else:
    # Select model
    model_choice = st.selectbox(
        "Select Model",
        ("gradient_boosting", "RandomForest")
    )

    # Forecast for Next 3days
    if st.button(f"Forecast Next 3days with {model_choice.capitalize()}", type="primary"):

        all_predictions = []
        model = models[model_choice]
        scaler = scalers[model_choice]

        with st.spinner("Fetching and predicting for next 3days..."):
            for hour_ahead in range(1, 73):  # 1 to 72 hours
                feature_row, display_data = get_single_forecast_data(hour_ahead, OWM_API_KEY)

                if feature_row:
                    X_scaled = scaler.transform([feature_row])
                    prediction = model.predict(X_scaled)[0]
                    display_data["Predicted AQI"] = round(prediction, 2)
                    all_predictions.append(display_data)

        if all_predictions:
            df_forecast = pd.DataFrame(all_predictions)

            # Display Results 
            st.success("3days AQI forecast generated successfully!")
            st.dataframe(df_forecast)

            # AQI Trend Chart 
            st.line_chart(
                df_forecast.set_index("Forecast Time")["Predicted AQI"],
                use_container_width=True
            )

            # Feature Importance 
            with st.expander("Show Feature Importance for this Model"):
                try:
                    importances = model.feature_importances_
                    df_fi = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': importances})
                    df_fi = df_fi.sort_values(by="Importance", ascending=False)
                    st.bar_chart(df_fi.set_index('Feature'))
                except Exception as e:
                    st.error(f"Could not load feature importance: {e}")

            # Raw Forecast Data
            with st.expander("Show Raw Forecast Data"):
                st.dataframe(df_forecast)

        else:
            st.error("Failed to generate forecast for the 72-hour period.")
