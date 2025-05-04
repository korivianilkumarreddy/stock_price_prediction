import streamlit as st
import helper10 as helper  # Updated to use helper10
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("📈 Stock Price Prediction App")

# Sidebar controls
with st.sidebar:
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.date.today())
    periods = st.selectbox("Prediction Days", [30, 90, 180, 365], index=1)
    prediction_date = st.date_input("Select Prediction Date", datetime.date.today() + datetime.timedelta(days=30))

if ticker:
    try:
        data = helper.fetch_stock_data(ticker, start_date, end_date)

        if not data.empty:
            close_column = next((col for col in data.columns if 'Close' in col), 'Close')

            # Historical data display
            st.subheader(f"📊 Historical Data for {ticker}")
            st.write(data.tail())

            # Historical price chart
            st.subheader("📉 Closing Price Trend")
            fig = px.line(data, x=data.index, y=close_column, title=f"{ticker} Stock Closing Prices")
            st.plotly_chart(fig)

            # Prophet prediction
            st.subheader("🔮 Future Price Prediction")
            model, forecast, rmse = helper.train_prophet_model(data, periods)  # Updated to include RMSE

            # Create interactive plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data[close_column], name='Actual Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Price',
                                     line=dict(color='orange', dash='dot')))
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))

            fig.update_layout(
                title=f'{ticker} Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified',
                showlegend=True
            )

            st.plotly_chart(fig)

            # Display RMSE
            st.subheader("📉 Model Performance")
            st.write(f"✅ RMSE (Root Mean Squared Error): `{rmse:.2f}`")

            # Specific date prediction with Profit/Loss calculation
            st.subheader("📅 Specific Date Prediction")
            try:
                selected_prediction = forecast[forecast['ds'] == pd.to_datetime(prediction_date)]
                actual_price = data[close_column].iloc[-1]  # Get the latest actual stock price

                if not selected_prediction.empty:
                    predicted_price = selected_prediction['yhat'].values[0]
                    profit_or_loss = round(predicted_price - actual_price, 2)

                    st.metric(
                        f"Predicted Price on {prediction_date}",
                        f"${predicted_price:.2f}",
                        delta=f"Uncertainty Range: {selected_prediction['yhat_lower'].values[0]:.2f} - {selected_prediction['yhat_upper'].values[0]:.2f}"
                    )

                    if profit_or_loss > 0:
                        st.success(f"📈 **Profit**: ${profit_or_loss:.2f} (Predicted > Actual)")
                    elif profit_or_loss < 0:
                        st.error(f"📉 **Loss**: ${abs(profit_or_loss):.2f} (Predicted < Actual)")
                    else:
                        st.info("⚖️ No Profit, No Loss (Prediction matches Actual Price)")
                else:
                    st.warning("Selected date is beyond prediction range. Try fewer prediction days.")
            except Exception as e:
                st.error(f"Error in date prediction: {str(e)}")

            # Explanation of How Data is Fetched
            st.subheader("📡 How Data is Fetched?")
            st.markdown("""
            - The app uses **Yahoo Finance API** (`yfinance`) to retrieve real-time and historical stock prices.
            - When you enter a stock ticker and date range, it sends a request to Yahoo Finance servers.
            - The response includes **Open, High, Low, Close, Volume, and Adjusted Prices**.
            - The Prophet model is then trained on this historical data to predict future prices.
            """)

        else:
            st.error("❌ No data found for this ticker. Try a different one!")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
