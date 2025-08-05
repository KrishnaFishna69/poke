#!/usr/bin/env python3
"""
Test script for Pokemon Price Forecasting System
"""

from forecasting import PokemonPriceForecaster
import logging

def test_forecasting():
    """Test the forecasting system"""
    logging.info("🧪 Testing Pokemon Price Forecasting System")
    
    # Initialize forecaster
    forecaster = PokemonPriceForecaster()
    
    # Test with cards that have price data
    test_cards = [
        "base1-1",  # Alakazam
        "base1-10", # Mewtwo
        "base1-100", # Lightning Energy
    ]
    
    for card_id in test_cards:
        logging.info(f"\n📊 Testing forecast for card: {card_id}")
        
        # Get forecast
        forecast = forecaster.forecast_card(card_id, horizon=7)
        
        if 'ensemble_forecast' in forecast:
            ensemble = forecast['ensemble_forecast']
            logging.info(f"✅ Forecast generated successfully!")
            logging.info(f"   Card: {forecast.get('card_name', 'Unknown')}")
            logging.info(f"   Last known price: ${forecast.get('last_known_price', 0):.2f}")
            logging.info(f"   Data points used: {forecast.get('data_points_used', 0)}")
            
            # Show 7-day prediction
            if ensemble['predictions']:
                pred_7day = ensemble['predictions'][-1]
                lower_7day = ensemble['lower_bound'][-1]
                upper_7day = ensemble['upper_bound'][-1]
                
                logging.info(f"   7-day prediction: ${pred_7day:.2f}")
                logging.info(f"   95% CI: ${lower_7day:.2f} - ${upper_7day:.2f}")
                
                # Calculate price change
                last_price = forecast.get('last_known_price', 0)
                if last_price > 0:
                    change_pct = ((pred_7day / last_price) - 1) * 100
                    logging.info(f"   Expected change: {change_pct:+.1f}%")
        else:
            logging.warning(f"❌ No forecast available for {card_id}")
    
    # Test summary function
    logging.info(f"\n📋 Testing forecast summary for {test_cards[0]}")
    summary = forecaster.get_forecast_summary(test_cards[0], horizon=7)
    
    if 'error' not in summary:
        logging.info(f"   Trend: {summary.get('trend', 'unknown')}")
        logging.info(f"   Volatility: {summary.get('volatility', 0):.2f}")
        logging.info(f"   Quality: {summary.get('forecast_quality', 'unknown')}")
    else:
        logging.warning(f"   {summary['error']}")

if __name__ == "__main__":
    test_forecasting() 