"""
Machine Learning Models for Food Waste Prediction
- LSTM (Long Short-Term Memory) for time series prediction
- Random Forest for tabular data prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Disable TensorFlow noise
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

class FoodWastePredictor:
    """Main class for food waste prediction using ML models"""
    
    def __init__(self):
        self.lstm_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.lstm_scaler = StandardScaler()
        self.sequence_length = 7  # Use 7 days of history for LSTM
        self.is_trained = False
        
    def prepare_data_for_lstm(self, df):
        """Prepare time series data for LSTM model"""
        df = df.copy()
        # Robust datetime parsing to handle ISO strings with or without time component
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Use FoodWaste as target, create sequences
        data = df['FoodWaste'].values.reshape(-1, 1)
        data_scaled = self.lstm_scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(data_scaled)):
            X.append(data_scaled[i-self.sequence_length:i, 0])
            y.append(data_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def prepare_data_for_rf(self, df):
        """Prepare tabular data for Random Forest model"""
        df = df.copy()
        # Robust datetime parsing to handle ISO strings with or without time component
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Feature engineering
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfMonth'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        
        # Encode MenuType
        if 'MenuType' in df.columns:
            df['MenuType_encoded'] = self.label_encoder.fit_transform(df['MenuType'])
        else:
            df['MenuType_encoded'] = 0
        
        # Create lag features
        df['FoodWaste_lag1'] = df['FoodWaste'].shift(1)
        df['FoodWaste_lag7'] = df['FoodWaste'].shift(7)
        df['Attendance_lag1'] = df['Attendance'].shift(1)
        
        # Rolling statistics
        df['FoodWaste_rolling_mean_7'] = df['FoodWaste'].rolling(window=7, min_periods=1).mean()
        df['FoodWaste_rolling_std_7'] = df['FoodWaste'].rolling(window=7, min_periods=1).std()
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Select features
        feature_cols = ['Attendance', 'MenuType_encoded', 'DayOfWeek', 'DayOfMonth', 'Month',
                       'FoodWaste_lag1', 'FoodWaste_lag7', 'Attendance_lag1',
                       'FoodWaste_rolling_mean_7', 'FoodWaste_rolling_std_7']
        
        X = df[feature_cols].values
        y = df['FoodWaste'].values
        
        return X, y, feature_cols
    
    def train_lstm(self, df, epochs=50, batch_size=16, validation_split=0.2):
        """Train LSTM model on historical data"""
        print("🔄 Training LSTM model...")
        
        X, y = self.prepare_data_for_lstm(df)
        
        if len(X) < 10:
            print("⚠️ Insufficient data for LSTM training. Need at least 10 samples.")
            return None
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        train_pred = model.predict(X_train, verbose=0)
        test_pred = model.predict(X_test, verbose=0)
        
        train_pred = self.lstm_scaler.inverse_transform(train_pred)
        test_pred = self.lstm_scaler.inverse_transform(test_pred)
        y_train_actual = self.lstm_scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = self.lstm_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        train_mae = mean_absolute_error(y_train_actual, train_pred)
        test_mae = mean_absolute_error(y_test_actual, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
        train_r2 = r2_score(y_train_actual, train_pred)
        test_r2 = r2_score(y_test_actual, test_pred)
        
        self.lstm_model = model
        
        metrics = {
            'train_mae': round(train_mae, 2),
            'test_mae': round(test_mae, 2),
            'train_rmse': round(train_rmse, 2),
            'test_rmse': round(test_rmse, 2),
            'train_r2': round(train_r2, 3),
            'test_r2': round(test_r2, 3),
            'epochs_trained': len(history.history['loss'])
        }
        
        print(f"✅ LSTM Training Complete!")
        print(f"   Train MAE: {metrics['train_mae']}, RMSE: {metrics['train_rmse']}, R²: {metrics['train_r2']}")
        print(f"   Test MAE: {metrics['test_mae']}, RMSE: {metrics['test_rmse']}, R²: {metrics['test_r2']}")
        
        return metrics
    
    def train_random_forest(self, df, n_estimators=100, max_depth=10, test_size=0.2):
        """Train Random Forest model on historical data"""
        print("🔄 Training Random Forest model...")
        
        X, y, feature_cols = self.prepare_data_for_rf(df)
        
        if len(X) < 10:
            print("⚠️ Insufficient data for Random Forest training. Need at least 10 samples.")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = rf_model.predict(X_train_scaled)
        test_pred = rf_model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
        
        self.rf_model = rf_model
        
        metrics = {
            'train_mae': round(train_mae, 2),
            'test_mae': round(test_mae, 2),
            'train_rmse': round(train_rmse, 2),
            'test_rmse': round(test_rmse, 2),
            'train_r2': round(train_r2, 3),
            'test_r2': round(test_r2, 3),
            'feature_importance': feature_importance
        }
        
        print(f"✅ Random Forest Training Complete!")
        print(f"   Train MAE: {metrics['train_mae']}, RMSE: {metrics['train_rmse']}, R²: {metrics['train_r2']}")
        print(f"   Test MAE: {metrics['test_mae']}, RMSE: {metrics['test_rmse']}, R²: {metrics['test_r2']}")
        
        return metrics
    
    def train_models(self, df, lstm_epochs=50, rf_n_estimators=100):
        """Train both LSTM and Random Forest models"""
        print("\n" + "="*50)
        print("🚀 Starting ML Model Training Process")
        print("="*50)
        
        lstm_metrics = self.train_lstm(df, epochs=lstm_epochs)
        rf_metrics = self.train_random_forest(df, n_estimators=rf_n_estimators)
        
        self.is_trained = True
        
        # Save models
        self.save_models()
        
        return {
            'lstm': lstm_metrics,
            'random_forest': rf_metrics
        }
    
    def predict_lstm(self, df, attendance=None, menu_type=None):
        """Predict using LSTM model"""
        if self.lstm_model is None:
            return None
        
        # Get last sequence_length days of data
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        if len(df) < self.sequence_length:
            return None
        
        # Use last sequence_length days
        recent_data = df['FoodWaste'].tail(self.sequence_length).values.reshape(-1, 1)
        recent_data_scaled = self.lstm_scaler.transform(recent_data)
        
        # Reshape for prediction
        X_input = recent_data_scaled.reshape(1, self.sequence_length, 1)
        
        # Predict
        prediction_scaled = self.lstm_model.predict(X_input, verbose=0)
        prediction = self.lstm_scaler.inverse_transform(prediction_scaled)[0][0]
        
        return max(50, min(300, prediction))
    
    def predict_random_forest(self, df, attendance, menu_type):
        """Predict using Random Forest model"""
        if self.rf_model is None:
            return None
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Create feature row for prediction
        last_row = df.iloc[-1]
        current_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)
        
        # Prepare features
        day_of_week = current_date.dayofweek
        day_of_month = current_date.day
        month = current_date.month
        
        # Encode menu type
        menu_types = ['Veg', 'Non-Veg', 'Special']
        if menu_type in menu_types:
            menu_encoded = menu_types.index(menu_type)
        else:
            menu_encoded = 0
        
        # Get lag features
        food_waste_lag1 = df['FoodWaste'].iloc[-1] if len(df) > 0 else 100
        food_waste_lag7 = df['FoodWaste'].iloc[-7] if len(df) >= 7 else food_waste_lag1
        attendance_lag1 = df['Attendance'].iloc[-1] if len(df) > 0 else attendance
        
        # Rolling statistics
        food_waste_rolling_mean_7 = df['FoodWaste'].tail(7).mean() if len(df) >= 7 else food_waste_lag1
        food_waste_rolling_std_7 = df['FoodWaste'].tail(7).std() if len(df) >= 7 else 10
        
        # Create feature array
        features = np.array([[
            attendance,
            menu_encoded,
            day_of_week,
            day_of_month,
            month,
            food_waste_lag1,
            food_waste_lag7,
            attendance_lag1,
            food_waste_rolling_mean_7,
            food_waste_rolling_std_7
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.rf_model.predict(features_scaled)[0]
        
        return max(50, min(300, prediction))
    
    def predict_ensemble(self, df, attendance, menu_type):
        """Ensemble prediction using both models"""
        lstm_pred = self.predict_lstm(df, attendance, menu_type)
        rf_pred = self.predict_random_forest(df, attendance, menu_type)
        
        if lstm_pred is None and rf_pred is None:
            return None
        elif lstm_pred is None:
            return rf_pred
        elif rf_pred is None:
            return lstm_pred
        else:
            # Weighted average (can be adjusted)
            return (lstm_pred * 0.5 + rf_pred * 0.5)
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            if self.lstm_model is not None:
                self.lstm_model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
            
            if self.rf_model is not None:
                with open(os.path.join(MODEL_DIR, 'rf_model.pkl'), 'wb') as f:
                    pickle.dump(self.rf_model, f)
            
            # Save scalers
            with open(os.path.join(MODEL_DIR, 'scalers.pkl'), 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'lstm_scaler': self.lstm_scaler,
                    'label_encoder': self.label_encoder
                }, f)
            
            print("💾 Models saved successfully!")
        except Exception as e:
            print(f"⚠️ Error saving models: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load LSTM
            lstm_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
            if os.path.exists(lstm_path):
                self.lstm_model = tf.keras.models.load_model(lstm_path)
            
            # Load Random Forest
            rf_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
            
            # Load scalers
            scalers_path = os.path.join(MODEL_DIR, 'scalers.pkl')
            if os.path.exists(scalers_path):
                with open(scalers_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler = scalers['scaler']
                    self.lstm_scaler = scalers['lstm_scaler']
                    self.label_encoder = scalers['label_encoder']
            
            self.is_trained = (self.lstm_model is not None) or (self.rf_model is not None)
            
            if self.is_trained:
                print("✅ Models loaded successfully!")
            else:
                print("⚠️ No saved models found. Train models first.")
            
            return self.is_trained
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            return False

