# 🌦️ Weather Nowcasting for Chennai using LSTM

This project focuses on weather nowcasting for Chennai, leveraging Long Short-Term Memory (LSTM) neural networks for time-series forecasting. The repository includes both univariate and multivariate forecasting approaches, implemented in Jupyter notebooks, as well as a Streamlit application for model testing.

## 📂 Repository Structure

- **`predictions.ipynb`**  
  Contains the implementation for **univariate forecasting**, using previous X days of weather data (e.g., temperature) as sequential input to predict future values.

- **`features.ipynb`**  
  Focuses on **multivariate forecasting**, incorporating multiple features like temperature, humidity, and other relevant weather parameters to improve the accuracy of predictions.

- **`app/test.py`**  
  A **Streamlit application** for testing the trained models. You can use this app to visualize and interact with real-time weather predictions.

## 🚀 Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/weather-nowcasting-lstm.git
    cd weather-nowcasting-lstm
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app to test models:
    ```bash
    streamlit run app/test.py
    ```

## Model Overview

- **Univariate LSTM**: Utilizes a single feature (like temperature) for predictions.
- **Multivariate LSTM**: Uses multiple weather features, such as temperature, humidity, wind speed, etc., to forecast future conditions more accurately.

