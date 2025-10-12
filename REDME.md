# 📈 AI Stock Price Predictor

A sophisticated stock price prediction system using LSTM Deep Learning with a beautiful Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)

---

## 🌟 Features

- **🤖 LSTM Neural Network** - 3-layer deep learning architecture
- **📊 23 Technical Indicators** - SMA, EMA, RSI, MACD, Bollinger Bands, and more
- **📈 Real-time Data** - Fetches live stock data from Yahoo Finance
- **🎨 Beautiful UI** - Interactive Streamlit dashboard
- **🔮 Multi-day Forecasts** - Predict 1-30 days ahead
- **📉 Performance Analytics** - Detailed model evaluation metrics
- **💾 Model Persistence** - Save and load trained models

---

## 📁 Project Structure

```
Stock_Price_Prediction/
│
├── app.py                          # Streamlit web application
├── enhanced_model_trainer.py       # Enhanced model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── Models/ (created after training)
│   ├── TSLA_best_model.keras      # Best model checkpoint
│   ├── TSLA_final_model.keras     # Final trained model
│   ├── TSLA_scaler.pkl            # Data scaler
│   └── TSLA_features.txt          # Feature list
│
└── Data/ (created during execution)
    └── historical_data.csv         # Downloaded stock data
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 2: Train a Model

```bash
# Train model for Tesla (or any stock)
python enhanced_model_trainer.py
```

**You'll be prompted to enter a stock symbol:**
```
Enter stock symbol (default: TSLA): AAPL
```

**Training takes 5-15 minutes depending on:**
- CPU/GPU availability
- Data size
- Number of epochs

### Step 3: Launch Streamlit App

```bash
# Start the web application
streamlit run app.py
```

**Your browser will open automatically at:** `http://localhost:8501`

---

## 📊 Model Architecture

### Enhanced LSTM Network

```
Input Layer (60 timesteps × 23 features)
    ↓
LSTM Layer 1 (100 units) + BatchNorm + Dropout(0.3)
    ↓
LSTM Layer 2 (100 units) + BatchNorm + Dropout(0.3)
    ↓
LSTM Layer 3 (100 units) + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (50 units, ReLU) + Dropout(0.3)
    ↓
Dense Layer (25 units, ReLU)
    ↓
Output Layer (1 unit, Linear)
```

**Total Parameters:** ~350,000

### Key Improvements over Basic Model

1. ✅ **More LSTM Units** (50 → 100) - Better pattern learning
2. ✅ **BatchNormalization** - Faster, more stable training
3. ✅ **Higher Dropout** (20% → 30%) - Reduced overfitting
4. ✅ **More Training Data** (5 → 7 years) - Better generalization
5. ✅ **Trend Features** - Captures momentum and direction
6. ✅ **70-15-15 Split** - Separate validation set

---

## 📈 Technical Indicators

The model uses 23 features:

### Price Data
- Open, High, Low, Close, Volume

### Moving Averages
- SMA (20, 50, 200 days)
- EMA (12, 26 days)

### Momentum Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- MACD Signal Line

### Volatility
- Bollinger Bands (Upper, Lower, Position)
- Volatility (10, 30 days)

### Trend Analysis
- 5-day, 10-day, 20-day momentum
- Volume ratio and trend

---

## 🎯 Using the App

### Main Interface

1. **Sidebar Configuration**
   - Enter stock symbol (e.g., AAPL, GOOGL, MSFT)
   - Select years of historical data (1-10)
   - Choose prediction horizon (1-30 days)

2. **Click "Run Prediction"**
   - App downloads data
   - Creates technical indicators
   - Loads trained model
   - Generates predictions

3. **View Results in Tabs:**
   - **📈 Prediction**: Future price forecasts
   - **📊 Technical Analysis**: RSI, MACD, Bollinger Bands
   - **🎯 Model Performance**: Accuracy metrics
   - **📋 Data**: Historical data table

### Key Metrics Explained

- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **RMSE (Root Mean Squared Error)**: Standard deviation of errors
- **R² Score**: % of variance explained (closer to 1 = better)
- **MAPE (Mean Absolute Percentage Error)**: Average % error
- **Accuracy**: 100 - MAPE

---

## 🔧 Advanced Usage

### Training Custom Models

```python
from enhanced_model_trainer import EnhancedStockPredictor

# Create predictor
predictor = EnhancedStockPredictor(
    stock_symbol='AAPL',
    years=10,              # More data
    sequence_length=90     # Longer lookback
)

# Train
df = predictor.download_data()
df_features = predictor.create_features(df)
(X_train, y_train), (X_val, y_val), (X_test, y_test), features = predictor.prepare_data(df_features)
model = predictor.build_model(n_features=len(features))
history = predictor.train(X_train, y_train, X_val, y_val, epochs=150)
results = predictor.evaluate(X_test, y_test)
```

### Supported Stocks

Works with any stock available on Yahoo Finance:

**Popular Examples:**
- Tech: AAPL, MSFT, GOOGL, META, NVDA
- Auto: TSLA, F, GM
- Finance: JPM, BAC, GS
- Retail: AMZN, WMT
- Index Funds: SPY, QQQ, DIA

---

## 📊 Performance Expectations

### Typical Results

| Metric | Good | Excellent |
|--------|------|-----------|
| R² Score | > 0.6 | > 0.8 |
| MAPE | < 15% | < 10% |
| MAE | < $10 | < $5 |

**Note:** Performance varies by:
- Stock volatility
- Market conditions
- Training data quality
- Prediction horizon

### Best Practices

1. **Train on stable periods** - Avoid training only on bull/bear markets
2. **Use sufficient data** - At least 5 years recommended
3. **Retrain regularly** - Monthly for active trading
4. **Validate assumptions** - Check if test period represents reality
5. **Don't overtrade** - Use as one of many decision factors

---

## ⚠️ Limitations & Disclaimers

### Model Limitations

- ❌ Cannot predict black swan events
- ❌ No news/sentiment analysis
- ❌ No fundamental data (earnings, P/E ratios)
- ❌ Assumes past patterns repeat
- ❌ Market conditions change

### Important Warnings

> **⚠️ NOT FINANCIAL ADVICE**
> 
> This tool is for **educational purposes only**. 
> 
> - Stock prediction is inherently uncertain
> - Past performance ≠ future results
> - Always consult financial advisors
> - Never invest more than you can afford to lose
> - Use predictions as ONE of MANY tools

---

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**
```
Error: Could not load model file
Solution: Run python enhanced_model_trainer.py first
```

**2. TensorFlow installation issues**
```
Error: DLL load failed
Solution: Install Visual C++ Redistributable
Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

**3. Stock symbol not found**
```
Error: Could not find data for XYZ
Solution: Check symbol is correct and available on Yahoo Finance
```

**4. Out of memory**
```
Error: ResourceExhaustedError
Solution: Reduce batch_size in training script
```

---

## 🔄 Model Updates

### When to Retrain

Retrain your model when:
- ✅ Market conditions change significantly
- ✅ Stock fundamentals change (mergers, splits)
- ✅ Model accuracy drops below 70%
- ✅ Monthly (for active trading)
- ✅ After major market events

### Quick Retrain

```bash
# Retrain with fresh data
python enhanced_model_trainer.py

# Enter symbol when prompted
# Model automatically overwrites old one
```

---

## 📚 Technical Details

### Data Pipeline

1. **Download** → Yahoo Finance API
2. **Clean** → Remove NaN, outliers
3. **Engineer** → Create 23 technical indicators
4. **Normalize** → MinMaxScaler (0-1)
5. **Sequence** → 60-day sliding windows
6. **Split** → 70% train, 15% val, 15% test

### Training Process

- **Optimizer:** Adam (lr=0.0005)
- **Loss:** Mean Squared Error
- **Callbacks:** Early Stopping, LR Reduction
- **Epochs:** 100 (typically stops at 30-40)
- **Batch Size:** 32
- **Validation:** Real-time on separate set

---

## 🤝 Contributing

Want to improve the model? Ideas:

1. Add sentiment analysis (news, Twitter)
2. Include fundamental data (P/E, revenue)
3. Ensemble multiple models
4. Add more technical indicators
5. Implement walk-forward validation

---

## 📞 Support

Having issues? Check:

1. ✅ Python 3.11 installed
2. ✅ All dependencies in requirements.txt
3. ✅ Model trained for the stock symbol
4. ✅ Internet connection (for data download)

---

## 📄 License

MIT License - Free for educational and personal use

---

## 🎓 Learning Resources

Want to understand the concepts better?

- **LSTM Networks:** [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **Technical Analysis:** [Investopedia TA Guide](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- **Time Series ML:** [Towards Data Science](https://towardsdatascience.com/)

---

## ✨ Acknowledgments

Built with:
- TensorFlow/Keras - Deep Learning
- Streamlit - Web Interface
- yfinance - Stock Data
- Plotly - Interactive Charts

---

**Made with ❤️ for learning and education**

*Remember: The best investment is in your own knowledge!* 📚