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
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
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

def get_all_card_ids():
    """Get all card IDs from the database."""
    conn = sqlite3.connect("pokemon_cards.db")
    try:
        df = pd.read_sql_query("SELECT id FROM price_history", conn)
        return df['id'].tolist()
    finally:
        conn.close()

def fetch_card_by_id(card_id, max_retries=3):
    """Fetch a specific card by ID with retries."""
    for attempt in range(max_retries):
        try:
            # Add delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 1.0))
            
            # Try to fetch the specific card
            cards = Card.where(id=card_id)
            if cards:
                return cards[0]
            
        except Exception as e:
            error_msg = str(e)
            if isinstance(e, bytes):
                try:
                    error_msg = e.decode('utf-8')
                except:
                    error_msg = "Unknown API error"
            
            logging.warning(f"Attempt {attempt + 1} failed for card {card_id}: {error_msg}")
            
            if attempt < max_retries - 1:
                time.sleep(random.uniform(2, 5))  # Longer delay between retries
    
    return None

def main():
    """Main function to update prices in the database."""
    logging.info("🚀 Starting COMPREHENSIVE price update")
    logging.info(f"Date: {today}")
    logging.info(f"Price column: {price_column}")
    
    # Check if database exists
    if not os.path.exists("pokemon_cards.db"):
        logging.error("❌ Database file not found!")
        sys.exit(1)
    
    # Add new price column to database
    add_price_column_to_database(price_column)
    
    # Get all card IDs from database
    all_card_ids = get_all_card_ids()
    logging.info(f"Found {len(all_card_ids)} cards in database")
    
    # Initialize counters
    cards_processed = 0
    cards_updated = 0
    cards_failed = 0
    cards_no_price = 0
    
    # Process each card individually
    logging.info("Processing cards individually for maximum coverage...")
    
    for i, card_id in enumerate(all_card_ids):
        try:
            # Log progress every 100 cards
            if i % 100 == 0:
                logging.info(f"Progress: {i}/{len(all_card_ids)} cards processed")
                logging.info(f"Updated: {cards_updated}, Failed: {cards_failed}, No Price: {cards_no_price}")
            
            # Fetch the specific card
            card = fetch_card_by_id(card_id)
            
            if card is None:
                logging.warning(f"Failed to fetch card {card_id}")
                cards_failed += 1
                # Set price to NULL to indicate no data
                update_card_price_in_database(card_id, None, price_column)
                continue
            
            # Get the market price
            try:
                market_price = get_market_price(card.tcgplayer.prices) if hasattr(card, 'tcgplayer') else None
                
                # Update price in database
                if update_card_price_in_database(card_id, market_price, price_column):
                    if market_price is not None:
                        cards_updated += 1
                        logging.debug(f"Updated: {safe_str(card.name)} - Price: ${market_price:.2f}")
                    else:
                        cards_no_price += 1
                        logging.debug(f"No price: {safe_str(card.name)}")
                
                cards_processed += 1
                
            except (AttributeError, TypeError) as e:
                logging.warning(f"Error getting price for {safe_str(card.name)}: {e}")
                update_card_price_in_database(card_id, None, price_column)
                cards_no_price += 1
                cards_processed += 1
                
        except Exception as e:
            logging.error(f"Error processing card {card_id}: {str(e)}")
            cards_failed += 1
            update_card_price_in_database(card_id, None, price_column)
            continue
    
    # Final statistics
    logging.info(f"✅ Price update completed!")
    logging.info(f"Total cards processed: {cards_processed}")
    logging.info(f"Cards with prices: {cards_updated}")
    logging.info(f"Cards without prices: {cards_no_price}")
    logging.info(f"Cards that failed: {cards_failed}")
    logging.info(f"Coverage: {(cards_updated/cards_processed)*100:.1f}%")
    
    # Show final statistics
    conn = sqlite3.connect("pokemon_cards.db")
    try:
        # Count cards with prices
        result = pd.read_sql_query(f"SELECT COUNT(*) as count FROM price_history WHERE '{price_column}' IS NOT NULL AND '{price_column}' > 0", conn)
        cards_with_prices = result['count'].iloc[0]
        
        # Get average price
        result = pd.read_sql_query(f"SELECT AVG('{price_column}') as avg_price FROM price_history WHERE '{price_column}' IS NOT NULL AND '{price_column}' > 0", conn)
        avg_price = result['avg_price'].iloc[0]
        
        logging.info(f"📊 Final Database Statistics:")
        logging.info(f"Cards with prices: {cards_with_prices}")
        logging.info(f"Average price: ${avg_price:.2f}")
        logging.info(f"Coverage: {cards_with_prices/len(all_card_ids)*100:.1f}%")
        
    finally:
        conn.close()
    
    # Exit with success
    logging.info("🎉 Comprehensive price update completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 