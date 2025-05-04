# 📈 Stock Price Prediction using Facebook Prophet

This project forecasts future stock prices using the Facebook Prophet model, a powerful tool for time series forecasting. It retrieves historical stock data via Yahoo Finance, applies Prophet for trend prediction, and displays the results using an interactive Streamlit app.
# 📈 Stock Price Prediction using Facebook Prophet

This project focuses on forecasting stock prices using the Facebook Prophet model, a robust and flexible library designed for time series prediction. It includes two Python files:

- `main.py`: This is the main Streamlit application that serves as the user interface. Users can input a stock ticker symbol and select a forecast period. The app then visualizes both historical data and future predictions in an interactive format.
  
- `helper.py`: This file contains the core logic of the application. It fetches historical stock data using the `yfinance` API, processes it into the required format, and applies the Prophet model to generate forecasts. The results are returned to the Streamlit interface for visualization.

The project demonstrates a practical implementation of time series forecasting in the financial domain. It combines live data extraction, statistical modeling, and web-based interactivity in a simple and effective way. Ideal for learners and enthusiasts interested in finance, data science, and machine learning.


## 🚀 How to Run

1. Make sure required libraries like `streamlit`, `prophet`, `yfinance`, and `pandas` are installed.
2. Run the app:
   ```bash
   streamlit run main10.py
🧾 Files
main10.py – Streamlit app for user interaction and displaying results.

helper10.py – Handles data fetching and applies the Prophet model.

📌 Note
This project is for learning and demonstration purposes only. Forecasts are based on historical data and should not be used for financial decisions.
