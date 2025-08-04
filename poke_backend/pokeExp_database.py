from pokemontcgsdk import Card
from pokemontcgsdk import Set
from pokemontcgsdk import Type
from pokemontcgsdk import Supertype
from pokemontcgsdk import Subtype
from pokemontcgsdk import Rarity
from pokemontcgsdk import RestClient
import pandas as pd
import sqlite3
import time
from datetime import datetime
import random
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('price_update.log'),
        logging.StreamHandler()
    ]
)

# Configure the API client
api_key = os.getenv('POKEMON_TCG_API_KEY', "85774024-7eef-4bc8-a628-3e3db902762e")
RestClient.configure(api_key)
logging.info("API client configured")

# Get today's date in the format MM-DD-YY
today = datetime.now().strftime("%m-%d-%y")
price_column = f"cardMarketPrice-{today}"

def safe_str(obj):
    """Safely convert an object to string, handling bytes and other special cases."""
    if obj is None:
        return None
    try:
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return str(obj)
    except Exception:
        return None

def safe_get(attr, default=None):
    """Safely get an attribute and convert it to string."""
    try:
        value = getattr(card, attr)
        if isinstance(value, (list, dict)):
            return value
        return safe_str(value)
    except (AttributeError, TypeError):
        return default

def get_market_price(prices):
    """Extract the market price from TCGPlayer prices."""
    if not prices:
        return None
    
    # Try to get the market price from any available price type
    for price_type in ['holofoil', 'reverseHolofoil', 'normal', 'firstEditionHolofoil', 'firstEditionNormal']:
        price_obj = getattr(prices, price_type, None)
        if price_obj and hasattr(price_obj, 'market') and price_obj.market is not None:
            return float(price_obj.market)
    
    return None

def add_price_column_to_database(price_column):
    """Add a new price column to the price_history table if it doesn't exist."""
    conn = sqlite3.connect("pokemon_cards.db")
    cursor = conn.cursor()
    
    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(price_history)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if price_column not in columns:
            cursor.execute(f"ALTER TABLE price_history ADD COLUMN '{price_column}' REAL")
            logging.info(f"Added new price column: {price_column}")
        else:
            logging.info(f"Price column {price_column} already exists")
            
    except Exception as e:
        logging.error(f"Error adding price column: {e}")
    finally:
        conn.close()

def update_card_price_in_database(card_id, price, price_column):
    """Update the price for a specific card in the database."""
    conn = sqlite3.connect("pokemon_cards.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"UPDATE price_history SET '{price_column}' = ? WHERE id = ?", (price, card_id))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error updating price for card {card_id}: {e}")
        return False
    finally:
        conn.close()

def get_existing_card_ids():
    """Get all card IDs from the database."""
    conn = sqlite3.connect("pokemon_cards.db")
    try:
        df = pd.read_sql_query("SELECT id FROM price_history", conn)
        return set(df['id'].tolist())
    finally:
        conn.close()

def main():
    """Main function to update prices in the database."""
    logging.info("Starting daily price update for database")
    
    # Add new price column to database
    add_price_column_to_database(price_column)
    
    # Get existing card IDs from database
    existing_card_ids = get_existing_card_ids()
    logging.info(f"Found {len(existing_card_ids)} existing cards in database")
    
    # Get all cards from API
    logging.info("Fetching all cards from API...")
    cards_processed = 0
    page = 1
    page_size = 250
    
    while True:
        logging.info(f"Fetching page {page}...")
        
        # Add delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 1.5))
        
        try:
            cards = Card.where(page=page, pageSize=page_size)
        except Exception as api_error:
            logging.error(f"API error on page {page}: {api_error}")
            logging.info("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        
        if not cards:
            logging.info("No more cards found, ending pagination.")
            break
            
        # Process cards for this page
        for card in cards:
            try:
                card_id = safe_str(getattr(card, 'id', None))
                if card_id is None:
                    continue
                
                # Only update cards that exist in our database
                if card_id in existing_card_ids:
                    # Get the market price
                    try:
                        market_price = get_market_price(card.tcgplayer.prices) if hasattr(card, 'tcgplayer') else None
                        
                        # Update price in database
                        if update_card_price_in_database(card_id, market_price, price_column):
                            logging.info(f"Updated: {safe_str(card.name)} - Price: {market_price}")
                            cards_processed += 1
                        
                        # Log progress every 100 cards
                        if cards_processed % 100 == 0:
                            logging.info(f"Processed {cards_processed} cards so far...")
                            
                    except (AttributeError, TypeError) as e:
                        logging.warning(f"Error getting price for {safe_str(card.name)}: {e}")
                        update_card_price_in_database(card_id, None, price_column)
                else:
                    logging.debug(f"Skipping {safe_str(card.name)} - not in database")
                    
            except Exception as e:
                logging.error(f"Error processing card: {str(e)}")
                continue
                
        page += 1
    
    logging.info(f"✅ Price update completed!")
    logging.info(f"Updated {cards_processed} cards with new price column: {price_column}")
    
    # Show some statistics
    conn = sqlite3.connect("pokemon_cards.db")
    try:
        # Count cards with prices
        result = pd.read_sql_query(f"SELECT COUNT(*) as count FROM price_history WHERE '{price_column}' IS NOT NULL", conn)
        cards_with_prices = result['count'].iloc[0]
        
        # Get average price
        result = pd.read_sql_query(f"SELECT AVG('{price_column}') as avg_price FROM price_history WHERE '{price_column}' IS NOT NULL", conn)
        avg_price = result['avg_price'].iloc[0]
        
        logging.info(f"Cards with prices: {cards_with_prices}")
        logging.info(f"Average price: ${avg_price:.2f}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main() 