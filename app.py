"""
STOCK PRICE PREDICTION APP
Beautiful Streamlit Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import keras
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 AI Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("### *Predict stock prices using LSTM Deep Learning*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Stock selection
    stock_symbol = st.text_input(
        "📊 Stock Symbol", 
        value="TSLA",
        help="Enter stock ticker (e.g., AAPL, GOOGL, MSFT)"
    ).upper()
    
    # Date range
    years = st.slider("📅 Years of Data", 1, 10, 5)
    
    # Prediction days
    pred_days = st.slider("🔮 Days to Predict", 1, 30, 7)
    
    # Action buttons
    run_prediction = st.button("🚀 Run Prediction", type="primary")
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    **Model:** LSTM Neural Network  
    **Features:** 23 technical indicators  
    **Training:** 70-15-15 split  
    **Framework:** TensorFlow/Keras  
    """)

# Main content area
if run_prediction:
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Download data
        status_text.text("📥 Downloading stock data...")
        progress_bar.progress(10)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        df = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            st.error(f"❌ Could not find data for {stock_symbol}. Please check the symbol.")
            st.stop()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        progress_bar.progress(30)
        
        # Step 2: Create features
        status_text.text("⚙️ Creating technical indicators...")
        
        # Technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['SMA_20']
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Trend features
        df['Trend_5'] = df['Close'].pct_change(5)
        df['Trend_10'] = df['Close'].pct_change(10)
        df['Trend_20'] = df['Close'].pct_change(20)
        
        # Volatility
        df['Volatility_10'] = df['Close'].rolling(10).std() / df['Close'].rolling(10).mean()
        df['Volatility_30'] = df['Close'].rolling(30).std() / df['Close'].rolling(30).mean()
        
        # Volume
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volume_Trend'] = df['Volume'].pct_change(5)
        
        df_clean = df.dropna()
        
        progress_bar.progress(50)
        
        # Step 3: Load model
        status_text.text("🤖 Loading AI model...")
        
        try:
            model = keras.models.load_model(f'{stock_symbol}_best_model.keras')
            scaler = joblib.load(f'{stock_symbol}_scaler.pkl')
            st.success(f"✅ Loaded pre-trained model for {stock_symbol}")
                
        except:
            st.warning("⚠️ No pre-trained model found. Using TSLA model as fallback...")
            try:
                model = keras.models.load_model('TSLA_best_model.keras')
                scaler = joblib.load('TSLA_scaler.pkl')
                st.info(f"Using TSLA model for {stock_symbol}. Results may be less accurate.")
            except:
                st.error("❌ No trained models found! Please train a model first.")
                st.code("python enhanced_model_trainer.py")
                st.stop()
        
        progress_bar.progress(70)
        
        # Step 4: Make predictions
        status_text.text("🔮 Making predictions...")
        
        # Prepare data
        feature_cols = [
            'Close', 'Open', 'High', 'Low', 'Volume',
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Position',
            'Trend_5', 'Trend_10', 'Trend_20',
            'Volatility_10', 'Volatility_30',
            'Volume_Ratio', 'Volume_Trend'
        ]
        
        data = df_clean[feature_cols].values
        data_scaled = scaler.transform(data)
        
        # Create last sequence
        sequence_length = 60
        last_sequence = data_scaled[-sequence_length:]
        last_sequence = last_sequence.reshape(1, sequence_length, len(feature_cols))
        
        # Predict next days
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(pred_days):
            next_pred = model.predict(current_sequence, verbose=0)[0][0]
            predictions.append(next_pred)
            
            new_row = current_sequence[0, -1, :].copy()
            new_row[0] = next_pred
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_row
        
        # Inverse transform
        n_features = len(feature_cols)
        def inverse_transform(values):
            dummy = np.zeros((len(values), n_features))
            dummy[:, 0] = values
            return scaler.inverse_transform(dummy)[:, 0]
        
        predictions_actual = inverse_transform(np.array(predictions))
        
        progress_bar.progress(90)
        
        # Step 5: Evaluate
        status_text.text("📊 Calculating metrics...")
        
        test_size = min(100, len(data_scaled) - sequence_length)
        X_test, y_test = [], []
        
        for i in range(len(data_scaled) - sequence_length - test_size, len(data_scaled) - sequence_length):
            X_test.append(data_scaled[i:i+sequence_length])
            y_test.append(data_scaled[i+sequence_length, 0])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        test_predictions = model.predict(X_test, verbose=0)
        
        y_test_actual = inverse_transform(y_test)
        test_predictions_actual = inverse_transform(test_predictions.flatten())
        
        mae = mean_absolute_error(y_test_actual, test_predictions_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions_actual))
        r2 = r2_score(y_test_actual, test_predictions_actual)
        mape = np.mean(np.abs((y_test_actual - test_predictions_actual) / y_test_actual)) * 100
        
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        
        # Display Results
        st.success(f"🎉 Prediction complete for **{stock_symbol}**!")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_price = df_clean['Close'].iloc[-1]
        predicted_price = predictions_actual[-1]
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric(
                f"Predicted ({pred_days}d)", 
                f"${predicted_price:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col3:
            st.metric("Accuracy", f"{100-mape:.2f}%")
        
        with col4:
            st.metric("MAE", f"${mae:.2f}")
        
        with col5:
            st.metric("R² Score", f"{r2:.4f}")
        
        st.markdown("---")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Prediction", "📊 Technical Analysis", "🎯 Performance"])
        
        with tab1:
            st.subheader("Stock Price Prediction")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_clean.index[-200:],
                y=df_clean['Close'].iloc[-200:],
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            future_dates = pd.date_range(
                start=df_clean.index[-1] + timedelta(days=1),
                periods=pred_days
            )
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions_actual,
                name='Predicted',
                line=dict(color='red', width=2, dash='dash'),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title=f'{stock_symbol} Price Forecast',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Detailed Predictions")
            pred_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': [f'${p:.2f}' for p in predictions_actual],
                'Change': [f'${p-current_price:+.2f}' for p in predictions_actual],
                'Change %': [f'{((p-current_price)/current_price)*100:+.2f}%' for p in predictions_actual]
            })
            st.dataframe(pred_df, use_container_width=True)
        
        with tab2:
            st.subheader("Technical Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rsi_val = df_clean['RSI'].iloc[-1]
                rsi_signal = "🔴 Overbought" if rsi_val > 70 else "🟢 Oversold" if rsi_val < 30 else "🟡 Neutral"
                st.metric("RSI", f"{rsi_val:.2f}", rsi_signal)
            
            with col2:
                macd_val = df_clean['MACD'].iloc[-1]
                macd_signal = "🟢 Bullish" if macd_val > df_clean['MACD_Signal'].iloc[-1] else "🔴 Bearish"
                st.metric("MACD", f"{macd_val:.2f}", macd_signal)
            
            with col3:
                bb_pos = df_clean['BB_Position'].iloc[-1]
                bb_signal = "🔴 Upper" if bb_pos > 0.8 else "🟢 Lower" if bb_pos < 0.2 else "🟡 Middle"
                st.metric("BB Position", f"{bb_pos:.2f}", bb_signal)
            
            with col4:
                vol = df_clean['Volatility_30'].iloc[-1]
                vol_signal = "🔴 High" if vol > 0.05 else "🟢 Low"
                st.metric("Volatility", f"{vol:.4f}", vol_signal)
            
            # Technical chart
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Price & Moving Averages', 'RSI'))
            
            fig.add_trace(go.Scatter(x=df_clean.index[-100:], y=df_clean['Close'].iloc[-100:],
                                    name='Close', line=dict(color='black')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_clean.index[-100:], y=df_clean['SMA_20'].iloc[-100:],
                                    name='SMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_clean.index[-100:], y=df_clean['SMA_50'].iloc[-100:],
                                    name='SMA 50'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df_clean.index[-100:], y=df_clean['RSI'].iloc[-100:],
                                    name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Model Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_test_actual, name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(y=test_predictions_actual, name='Predicted', line=dict(color='red')))
                fig.update_layout(title='Actual vs Predicted', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                errors = test_predictions_actual - y_test_actual
                fig = go.Figure(data=[go.Histogram(x=errors, nbinsx=30)])
                fig.update_layout(title='Error Distribution', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'R² Score', 'MAPE', 'Accuracy'],
                'Value': [f'${mae:.2f}', f'${rmse:.2f}', f'{r2:.4f}', f'{mape:.2f}%', f'{100-mape:.2f}%']
            })
            st.table(metrics_df)
        
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.markdown("""
    ## 👋 Welcome to AI Stock Price Predictor!
    
    ### 🚀 How to Use:
    1. Enter a stock symbol in the sidebar
    2. Select parameters
    3. Click "Run Prediction"
    
    ### ⚠️ First Time Setup:
    Train a model first:
    ```python
    python enhanced_model_trainer.py
    ```
    
    ### 📈 Popular Stocks:
    TSLA, AAPL, GOOGL, MSFT, AMZN, META, NVDA
    """)
    
    st.info("⚠️ **Disclaimer:** For educational purposes only. Not financial advice.")