#!/usr/bin/env python3
"""
Probabilistic Pokemon Card Price Forecasting System
Predicts 7-day price forecasts with confidence intervals using historical data
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import pickle
import os
from pathlib import Path

# Forecasting libraries
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forecasting.log'),
        logging.StreamHandler()
    ]
)

class PokemonPriceForecaster:
    """
    Probabilistic price forecasting system for Pokemon cards
    """
    
    def __init__(self, db_path: str = "pokemon_cards.db", models_dir: str = "forecast_models"):
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.card_metadata = {}
        
        # Forecasting parameters
        self.forecast_horizon = 7  # 7 days
        self.confidence_level = 0.95  # 95% confidence intervals
        
    def get_price_history(self, card_id: str) -> pd.DataFrame:
        """
        Extract price history for a specific card from the database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all price columns
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(price_history)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Filter for price columns (cardMarketPrice-*)
            price_columns = [col for col in columns if col.startswith('cardMarketPrice-')]
            
            if not price_columns:
                logging.warning(f"No price columns found for card {card_id}")
                return pd.DataFrame()
            
            # Get price data for this card
            quoted_columns = [f'"{col}"' for col in price_columns]
            query = f"""
                SELECT id, {', '.join(quoted_columns)}
                FROM price_history 
                WHERE id = ?
            """
            
            df = pd.read_sql_query(query, conn, params=[card_id])
            conn.close()
            
            if df.empty:
                logging.warning(f"No price data found for card {card_id}")
                return pd.DataFrame()
            
            # Transpose to get dates as index
            df_transposed = df.T
            df_transposed.columns = ['price']
            df_transposed = df_transposed.iloc[1:]  # Remove 'id' row
            
            # Convert index to datetime, filtering out test/simulation columns
            date_strings = df_transposed.index.str.replace('cardMarketPrice-', '')
            # Filter out test/simulation columns
            valid_dates = []
            valid_indices = []
            
            for i, date_str in enumerate(date_strings):
                if not any(suffix in date_str for suffix in ['-test', '-simulation']):
                    try:
                        pd.to_datetime(date_str, format='%m-%d-%y')
                        valid_dates.append(date_str)
                        valid_indices.append(i)
                    except:
                        continue
            
            if not valid_dates:
                logging.warning(f"No valid price dates found for card {card_id}")
                return pd.DataFrame()
            
            # Keep only valid columns
            df_transposed = df_transposed.iloc[valid_indices]
            df_transposed.index = pd.to_datetime(valid_dates, format='%m-%d-%y')
            
            # Sort by date
            df_transposed = df_transposed.sort_index()
            
            # Remove rows with no price data
            df_transposed = df_transposed.dropna()
            
            # Reset index to get date as column
            df_transposed = df_transposed.reset_index()
            df_transposed.columns = ['ds', 'y']  # Prophet format
            
            return df_transposed
            
        except Exception as e:
            logging.error(f"Error getting price history for card {card_id}: {e}")
            return pd.DataFrame()
    
    def get_card_info(self, card_id: str) -> Dict:
        """
        Get card information from database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT id, name, supertype, subtypes, types, hp, 
                       attacks, abilities, weaknesses, resistances
                FROM cards 
                WHERE id = ?
            """
            df = pd.read_sql_query(query, conn, params=[card_id])
            conn.close()
            
            if not df.empty:
                return df.iloc[0].to_dict()
            return {}
            
        except Exception as e:
            logging.error(f"Error getting card info for {card_id}: {e}")
            return {}
    
    def preprocess_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess price data for forecasting
        """
        if df.empty:
            return df
        
        # Remove outliers (prices that are too high or too low)
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean = df[(df['y'] >= lower_bound) & (df['y'] <= upper_bound)]
        
        # Ensure minimum data points
        if len(df_clean) < 10:
            logging.warning(f"Insufficient data points ({len(df_clean)}) for reliable forecasting")
            return df_clean
        
        # Add trend and seasonality features
        df_clean['trend'] = np.arange(len(df_clean))
        df_clean['day_of_week'] = df_clean['ds'].dt.dayofweek
        
        return df_clean
    
    def train_prophet_model(self, card_id: str, price_data: pd.DataFrame) -> Optional[Prophet]:
        """
        Train Prophet model for a specific card
        """
        if price_data.empty or len(price_data) < 10:
            logging.warning(f"Insufficient data for card {card_id}")
            return None
        
        try:
            # Configure Prophet model
            model = Prophet(
                yearly_seasonality=False,  # Not enough data for yearly patterns
                weekly_seasonality=True,   # Weekly patterns in card prices
                daily_seasonality=False,   # No daily patterns
                seasonality_mode='multiplicative',  # Price changes are multiplicative
                changepoint_prior_scale=0.05,  # Allow for trend changes
                seasonality_prior_scale=10.0,  # Regularize seasonality
                interval_width=self.confidence_level
            )
            
            # Add custom seasonality for weekends (card prices might differ)
            model.add_seasonality(name='weekly', period=7, fourier_order=3)
            
            # Fit the model
            model.fit(price_data)
            
            return model
            
        except Exception as e:
            logging.error(f"Error training Prophet model for card {card_id}: {e}")
            return None
    
    def train_ensemble_model(self, card_id: str, price_data: pd.DataFrame) -> Dict:
        """
        Train ensemble of forecasting models for better uncertainty estimation
        """
        if price_data.empty or len(price_data) < 10:
            return {}
        
        try:
            models = {}
            
            # 1. Prophet model
            prophet_model = self.train_prophet_model(card_id, price_data)
            if prophet_model:
                models['prophet'] = prophet_model
            
            # 2. Simple moving average model
            if len(price_data) >= 7:
                ma_model = {
                    'type': 'moving_average',
                    'window': 7,
                    'last_prices': price_data['y'].tail(7).tolist()
                }
                models['ma'] = ma_model
            
            # 3. Linear trend model
            if len(price_data) >= 5:
                try:
                    x = np.arange(len(price_data))
                    y = price_data['y'].astype(float).values
                    coeffs = np.polyfit(x, y, 1)
                    trend_model = {
                        'type': 'linear_trend',
                        'slope': float(coeffs[0]),
                        'intercept': float(coeffs[1]),
                        'last_date': price_data['ds'].iloc[-1],
                        'last_price': float(price_data['y'].iloc[-1])
                    }
                    models['trend'] = trend_model
                except Exception as e:
                    logging.warning(f"Linear trend model failed for card {card_id}: {e}")
            
            return models
            
        except Exception as e:
            logging.error(f"Error training ensemble model for card {card_id}: {e}")
            return {}
    
    def forecast_card(self, card_id: str, horizon: int = 7) -> Dict:
        """
        Forecast prices for a specific card
        
        Args:
            card_id: Card identifier
            horizon: Number of days to forecast (default: 7)
            
        Returns:
            Dictionary with forecast data including predictions and confidence intervals
        """
        try:
            # Get card info
            card_info = self.get_card_info(card_id)
            if not card_info:
                return self._empty_forecast_result(card_id, horizon)
            
            # Get price history
            price_data = self.get_price_history(card_id)
            if price_data.empty:
                return self._empty_forecast_result(card_id, horizon)
            
            # Preprocess data
            price_data = self.preprocess_price_data(price_data)
            if len(price_data) < 10:
                return self._empty_forecast_result(card_id, horizon)
            
            # Check if model exists, otherwise train
            model_path = self.models_dir / f"{card_id}_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[card_id] = pickle.load(f)
            else:
                self.models[card_id] = self.train_ensemble_model(card_id, price_data)
                # Save model
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models[card_id], f)
            
            if not self.models[card_id]:
                logging.warning(f"No models trained for {card_id}, returning empty result")
                return self._empty_forecast_result(card_id, horizon)
            
            # Generate forecasts
            forecasts = self._generate_ensemble_forecast(card_id, price_data, horizon)
            
            # Add metadata
            forecasts.update({
                'card_id': card_id,
                'card_name': card_info.get('name', 'Unknown'),
                'last_known_price': float(price_data['y'].iloc[-1]),
                'last_known_date': price_data['ds'].iloc[-1].strftime('%Y-%m-%d'),
                'data_points_used': len(price_data),
                'forecast_generated_at': datetime.now().isoformat()
            })
            
            return forecasts
            
        except Exception as e:
            logging.error(f"Error forecasting card {card_id}: {e}")
            return self._empty_forecast_result(card_id, horizon)
    
    def _generate_ensemble_forecast(self, card_id: str, price_data: pd.DataFrame, horizon: int) -> Dict:
        """
        Generate ensemble forecast using multiple models
        """
        forecasts = {}
        predictions = []
        
        models = self.models[card_id]
        
        # Generate future dates
        last_date = price_data['ds'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=horizon, freq='D')
        
        # Prophet forecast
        if 'prophet' in models:
            try:
                future_df = pd.DataFrame({'ds': future_dates})
                prophet_forecast = models['prophet'].predict(future_df)
                
                prophet_pred = {
                    'model': 'prophet',
                    'predictions': prophet_forecast['yhat'].tolist(),
                    'lower_bound': prophet_forecast['yhat_lower'].tolist(),
                    'upper_bound': prophet_forecast['yhat_upper'].tolist(),
                    'dates': future_dates.strftime('%Y-%m-%d').tolist()
                }
                predictions.append(prophet_pred)
                
            except Exception as e:
                logging.warning(f"Prophet forecast failed for card {card_id}: {e}")
        
        # Moving average forecast
        if 'ma' in models:
            try:
                ma_model = models['ma']
                last_prices = ma_model['last_prices']
                ma_prediction = np.mean(last_prices)
                
                # Simple uncertainty estimation
                price_std = np.std(last_prices)
                ma_pred = {
                    'model': 'moving_average',
                    'predictions': [ma_prediction] * horizon,
                    'lower_bound': [max(0, ma_prediction - 1.96 * price_std)] * horizon,
                    'upper_bound': [ma_prediction + 1.96 * price_std] * horizon,
                    'dates': future_dates.strftime('%Y-%m-%d').tolist()
                }
                predictions.append(ma_pred)
                
            except Exception as e:
                logging.warning(f"MA forecast failed for card {card_id}: {e}")
        
        # Linear trend forecast
        if 'trend' in models:
            try:
                trend_model = models['trend']
                slope = trend_model['slope']
                intercept = trend_model['intercept']
                last_date = trend_model['last_date']
                last_price = trend_model['last_price']
                
                trend_predictions = []
                for i in range(1, horizon + 1):
                    days_ahead = i
                    trend_price = last_price + (slope * days_ahead)
                    trend_predictions.append(max(0, trend_price))
                
                # Uncertainty based on historical volatility
                price_volatility = price_data['y'].pct_change().std()
                trend_pred = {
                    'model': 'linear_trend',
                    'predictions': trend_predictions,
                    'lower_bound': [max(0, p * (1 - 1.96 * price_volatility)) for p in trend_predictions],
                    'upper_bound': [p * (1 + 1.96 * price_volatility) for p in trend_predictions],
                    'dates': future_dates.strftime('%Y-%m-%d').tolist()
                }
                predictions.append(trend_pred)
                
            except Exception as e:
                logging.warning(f"Trend forecast failed for card {card_id}: {e}")
        
        # Combine predictions
        if predictions:
            ensemble_forecast = self._combine_predictions(predictions, future_dates)
            forecasts['ensemble_forecast'] = ensemble_forecast
            forecasts['individual_models'] = predictions
        
        return forecasts
    
    def _combine_predictions(self, predictions: List[Dict], future_dates: pd.DatetimeIndex) -> Dict:
        """
        Combine predictions from multiple models using weighted averaging
        """
        if not predictions:
            return {}
        
        # Weights for different models (can be tuned based on performance)
        weights = {
            'prophet': 0.5,
            'moving_average': 0.3,
            'linear_trend': 0.2
        }
        
        combined_predictions = []
        combined_lower = []
        combined_upper = []
        
        for i in range(len(future_dates)):
            weighted_pred = 0
            weighted_lower = 0
            weighted_upper = 0
            total_weight = 0
            
            for pred in predictions:
                model_name = pred['model']
                weight = weights.get(model_name, 0.1)
                
                if i < len(pred['predictions']):
                    weighted_pred += pred['predictions'][i] * weight
                    weighted_lower += pred['lower_bound'][i] * weight
                    weighted_upper += pred['upper_bound'][i] * weight
                    total_weight += weight
            
            if total_weight > 0:
                combined_predictions.append(weighted_pred / total_weight)
                combined_lower.append(weighted_lower / total_weight)
                combined_upper.append(weighted_upper / total_weight)
            else:
                combined_predictions.append(0)
                combined_lower.append(0)
                combined_upper.append(0)
        
        return {
            'predictions': combined_predictions,
            'lower_bound': combined_lower,
            'upper_bound': combined_upper,
            'confidence_level': self.confidence_level,
            'dates': future_dates.strftime('%Y-%m-%d').tolist()
        }
    
    def _empty_forecast_result(self, card_id: str, horizon: int) -> Dict:
        """
        Return empty forecast result when insufficient data
        """
        future_dates = pd.date_range(start=datetime.now() + timedelta(days=1), 
                                   periods=horizon, freq='D')
        
        return {
            'card_id': card_id,
            'card_name': 'Unknown',
            'ensemble_forecast': {
                'predictions': [0] * horizon,
                'lower_bound': [0] * horizon,
                'upper_bound': [0] * horizon,
                'confidence_level': self.confidence_level,
                'dates': future_dates.strftime('%Y-%m-%d').tolist()
            },
            'last_known_price': 0,
            'last_known_date': datetime.now().strftime('%Y-%m-%d'),
            'data_points_used': 0,
            'forecast_generated_at': datetime.now().isoformat(),
            'error': 'Insufficient price data for forecasting'
        }
    
    def forecast_multiple_cards(self, card_ids: List[str], horizon: int = 7) -> Dict:
        """
        Forecast prices for multiple cards
        """
        results = {}
        
        for card_id in card_ids:
            logging.info(f"Forecasting card: {card_id}")
            forecast = self.forecast_card(card_id, horizon)
            results[card_id] = forecast
        
        return results
    
    def get_forecast_summary(self, card_id: str, horizon: int = 7) -> Dict:
        """
        Get a summary of the forecast for a card
        """
        forecast = self.forecast_card(card_id, horizon)
        
        if 'ensemble_forecast' not in forecast:
            return {'error': 'No forecast available'}
        
        ensemble = forecast['ensemble_forecast']
        
        # Calculate summary statistics
        predictions = ensemble['predictions']
        lower_bounds = ensemble['lower_bound']
        upper_bounds = ensemble['upper_bound']
        
        summary = {
            'card_id': card_id,
            'card_name': forecast.get('card_name', 'Unknown'),
            'forecast_horizon': horizon,
            'last_known_price': forecast.get('last_known_price', 0),
            'predicted_price_7_days': predictions[-1] if predictions else 0,
            'price_change_7_days': (predictions[-1] - forecast.get('last_known_price', 0)) if predictions else 0,
            'price_change_percent_7_days': ((predictions[-1] / forecast.get('last_known_price', 1) - 1) * 100) if predictions and forecast.get('last_known_price', 0) > 0 else 0,
            'confidence_interval_7_days': {
                'lower': lower_bounds[-1] if lower_bounds else 0,
                'upper': upper_bounds[-1] if upper_bounds else 0
            },
            'trend': 'increasing' if predictions and predictions[-1] > forecast.get('last_known_price', 0) else 'decreasing' if predictions and predictions[-1] < forecast.get('last_known_price', 0) else 'stable',
            'volatility': np.std(predictions) if predictions else 0,
            'forecast_quality': 'high' if forecast.get('data_points_used', 0) >= 30 else 'medium' if forecast.get('data_points_used', 0) >= 15 else 'low'
        }
        
        return summary

def main():
    """
    Example usage and testing of the forecasting system
    """
    logging.info("🚀 Starting Pokemon Price Forecasting System")
    
    # Initialize forecaster
    forecaster = PokemonPriceForecaster()
    
    # Get some cards with price data
    try:
        conn = sqlite3.connect("pokemon_cards.db")
        
        # Find cards with recent price data
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(price_history)")
        columns = [col[1] for col in cursor.fetchall()]
        price_columns = [col for col in columns if col.startswith('cardMarketPrice-')]
        
        if not price_columns:
            logging.error("No price data found in database")
            return
        
        # Get cards with recent prices
        recent_price_col = price_columns[-1]  # Most recent price column
        query = f"""
            SELECT id, name 
            FROM card_info ci
            JOIN price_history ph ON ci.id = ph.id
            WHERE ph."{recent_price_col}" IS NOT NULL 
            AND ph."{recent_price_col}" > 0
            LIMIT 5
        """
        
        test_cards = pd.read_sql_query(query, conn)
        conn.close()
        
        if test_cards.empty:
            logging.error("No cards with recent price data found")
            return
        
        logging.info(f"Testing forecasting with {len(test_cards)} cards")
        
        # Test forecasting for each card
        for _, card in test_cards.iterrows():
            card_id = card['id']
            card_name = card['name']
            
            logging.info(f"Forecasting: {card_name} ({card_id})")
            
            # Get forecast
            forecast = forecaster.forecast_card(card_id, horizon=7)
            
            # Get summary
            summary = forecaster.get_forecast_summary(card_id, horizon=7)
            
            logging.info(f"  Last known price: ${summary.get('last_known_price', 0):.2f}")
            logging.info(f"  7-day prediction: ${summary.get('predicted_price_7_days', 0):.2f}")
            logging.info(f"  Price change: {summary.get('price_change_percent_7_days', 0):.1f}%")
            logging.info(f"  Trend: {summary.get('trend', 'unknown')}")
            logging.info(f"  Quality: {summary.get('forecast_quality', 'unknown')}")
            
            if 'confidence_interval_7_days' in summary:
                ci = summary['confidence_interval_7_days']
                logging.info(f"  95% CI: ${ci['lower']:.2f} - ${ci['upper']:.2f}")
            
            logging.info("  " + "-" * 50)
        
        logging.info("✅ Forecasting test completed!")
        
    except Exception as e:
        logging.error(f"Error in main test: {e}")

if __name__ == "__main__":
    main() 