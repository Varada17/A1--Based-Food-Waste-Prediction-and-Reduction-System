# Machine Learning Models for Food Waste Prediction

This project now includes **LSTM** (Long Short-Term Memory) and **Random Forest** machine learning models for food waste prediction.

## Features

### 🤖 ML Models

1. **LSTM Model** - Time series prediction using deep learning
   - Uses 7 days of historical data as input sequence
   - 2-layer LSTM architecture with dropout regularization
   - Optimized for temporal patterns in food waste data

2. **Random Forest Model** - Tabular data prediction
   - Feature engineering: lag features, rolling statistics, date features
   - Handles attendance, menu type, and historical patterns
   - Provides feature importance insights

3. **Ensemble Prediction** - Combines both models
   - Weighted average of LSTM and Random Forest predictions
   - More robust and accurate predictions

## Training Process

### How to Train Models

1. **Start the Dashboard**
   ```bash
   python app.py
   ```

2. **Train Models**
   - Click the **"🤖 TRAIN ML MODELS"** button in the dashboard
   - The system will:
     - Load historical data
     - Preprocess data for both models
     - Train LSTM with early stopping
     - Train Random Forest with feature engineering
     - Evaluate both models (MAE, RMSE, R²)
     - Save models to `models/` directory

3. **Use Trained Models**
   - After training, predictions automatically use ML models
   - If models aren't trained, falls back to rule-based forecasting

### Model Training Details

#### LSTM Training
- **Input**: 7-day sequences of food waste data
- **Architecture**: 
  - LSTM layer (50 units) → Dropout (0.2)
  - LSTM layer (50 units) → Dropout (0.2)
  - Dense layer (25 units)
  - Output layer (1 unit)
- **Optimizer**: Adam
- **Loss**: Mean Squared Error
- **Early Stopping**: Monitors validation loss with patience=10
- **Epochs**: Up to 50 (early stopping may stop earlier)

#### Random Forest Training
- **Features**:
  - Attendance
  - Menu Type (encoded)
  - Day of Week, Day of Month, Month
  - Lag features (1-day, 7-day)
  - Rolling statistics (mean, std over 7 days)
- **Parameters**:
  - n_estimators: 100
  - max_depth: 10
  - Random state: 42
- **Train/Test Split**: 80/20

### Model Evaluation Metrics

Both models are evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (R-squared / Coefficient of Determination)

Metrics are calculated separately for training and test sets.

## File Structure

```
├── app.py                 # Main Dash application
├── ml_models.py           # ML model implementations
├── requirements.txt       # Python dependencies
├── models/                # Saved model files (created after training)
│   ├── lstm_model.h5      # Trained LSTM model
│   ├── rf_model.pkl       # Trained Random Forest model
│   └── scalers.pkl        # Data scalers and encoders
└── food_waste_predictions.json  # Historical data storage
```

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `tensorflow` / `keras` - For LSTM model
- `scikit-learn` - For Random Forest and preprocessing
- `pandas`, `numpy` - Data manipulation
- `dash`, `plotly` - Dashboard interface

## Usage Workflow

1. **Initial Setup**: Historical data is automatically loaded/created
2. **Train Models**: Click "TRAIN ML MODELS" button
3. **Make Predictions**: 
   - Select menu type
   - Enter expected attendance
   - Click "PREDICT & SAVE"
4. **View Results**: 
   - See predictions from LSTM, Random Forest, and Ensemble
   - View model performance metrics
   - Check prediction history

## Model Persistence

- Trained models are automatically saved to `models/` directory
- Models are loaded automatically when the app starts
- You can retrain models anytime by clicking the train button again

## Fallback Behavior

- If models aren't trained: Uses rule-based forecasting
- If ML prediction fails: Falls back to rule-based forecasting
- Ensures the system always provides predictions

## Performance Tips

- **More Data = Better Models**: Ensure you have at least 30+ days of historical data
- **Regular Retraining**: Retrain models periodically as new data accumulates
- **Feature Engineering**: Random Forest automatically creates lag and rolling features
- **Early Stopping**: LSTM uses early stopping to prevent overfitting

## Troubleshooting

**"Insufficient data for training"**
- Need at least 10 samples (days) of historical data
- More data (30+ days) recommended for better performance

**"Models Not Trained Yet"**
- Click "TRAIN ML MODELS" button first
- Check that historical data exists in `food_waste_predictions.json`

**Import Errors**
- Run `pip install -r requirements.txt`
- Ensure TensorFlow is properly installed

## Future Enhancements

Potential improvements:
- Hyperparameter tuning interface
- Model comparison charts
- Feature importance visualization
- Automated retraining schedule
- Additional ML models (XGBoost, Prophet, etc.)

