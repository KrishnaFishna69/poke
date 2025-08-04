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

def get_existing_card_ids():
    """Get all card IDs from the database."""
    conn = sqlite3.connect("pokemon_cards.db")
    try:
        df = pd.read_sql_query("SELECT id FROM price_history", conn)
        return set(df['id'].tolist())
    finally:
        conn.close()

def main():
    """Main function to update prices using batch processing."""
    logging.info("🚀 Starting BATCH price update")
    logging.info(f"Date: {today}")
    logging.info(f"Price column: {price_column}")
    
    # Check if database exists
    if not os.path.exists("pokemon_cards.db"):
        logging.error("❌ Database file not found!")
        sys.exit(1)
    
    # Add new price column to database
    add_price_column_to_database(price_column)
    
    # Get existing card IDs from database
    existing_card_ids = get_existing_card_ids()
    logging.info(f"Found {len(existing_card_ids)} existing cards in database")
    
    # Get all cards from API using batch processing
    logging.info("Fetching all cards from API using batch processing...")
    cards_processed = 0
    cards_updated = 0
    cards_no_price = 0
    page = 1
    page_size = 250
    max_pages = 100  # Safety limit
    
    while page <= max_pages:
        logging.info(f"Fetching page {page}...")
        
        # Add delay to avoid rate limiting
        time.sleep(random.uniform(1.0, 2.0))
        
        try:
            cards = Card.where(page=page, pageSize=page_size)
        except Exception as api_error:
            error_msg = str(api_error)
            if isinstance(api_error, bytes):
                try:
                    error_msg = api_error.decode('utf-8')
                except:
                    error_msg = "Unknown API error"
            
            logging.error(f"API error on page {page}: {error_msg}")
            logging.info("Retrying in 10 seconds...")
            time.sleep(10)
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
                            if market_price is not None:
                                cards_updated += 1
                                logging.debug(f"Updated: {safe_str(card.name)} - Price: ${market_price:.2f}")
                            else:
                                cards_no_price += 1
                                logging.debug(f"No price: {safe_str(card.name)}")
                        
                        cards_processed += 1
                        
                        # Log progress every 50 cards
                        if cards_processed % 50 == 0:
                            logging.info(f"Processed {cards_processed} cards, updated {cards_updated} prices...")
                            
                    except (AttributeError, TypeError) as e:
                        logging.warning(f"Error getting price for {safe_str(card.name)}: {e}")
                        update_card_price_in_database(card_id, None, price_column)
                        cards_no_price += 1
                        cards_processed += 1
                else:
                    logging.debug(f"Skipping {safe_str(card.name)} - not in database")
                    
            except Exception as e:
                logging.error(f"Error processing card: {str(e)}")
                continue
                
        page += 1
    
    # Final statistics
    logging.info(f"✅ Batch price update completed!")
    logging.info(f"Total cards processed: {cards_processed}")
    logging.info(f"Cards with prices: {cards_updated}")
    logging.info(f"Cards without prices: {cards_no_price}")
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
        logging.info(f"Coverage: {cards_with_prices/len(existing_card_ids)*100:.1f}%")
        
    finally:
        conn.close()
    
    # Exit with success
    logging.info("🎉 Batch price update completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 