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
import json
import shutil

# Set up comprehensive logging
def setup_logging():
    """Setup comprehensive logging for GitHub Actions"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/price_update_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Configure the API client
api_key = os.getenv('POKEMON_TCG_API_KEY', "85774024-7eef-4bc8-a628-3e3db902762e")
RestClient.configure(api_key)

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
            logging.info(f"✅ Added new price column: {price_column}")
        else:
            logging.info(f"ℹ️ Price column {price_column} already exists")
            
    except Exception as e:
        logging.error(f"❌ Error adding price column: {e}")
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
        logging.error(f"❌ Error updating price for card {card_id}: {e}")
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

def create_progress_file(progress_data):
    """Create a progress file for monitoring"""
    progress_file = "logs/update_progress.json"
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def make_database_accessible():
    """Make database accessible to frontend scripts"""
    try:
        # Copy database to frontend directory
        frontend_db_path = "../poke_frontend/pokemon_cards.db"
        shutil.copy2("pokemon_cards.db", frontend_db_path)
        logging.info(f"✅ Database copied to frontend: {frontend_db_path}")
        
        # Create a symlink in the root directory for easy access
        root_db_path = "../pokemon_cards.db"
        if os.path.exists(root_db_path):
            os.remove(root_db_path)
        shutil.copy2("pokemon_cards.db", root_db_path)
        logging.info(f"✅ Database copied to root: {root_db_path}")
        
        return True
    except Exception as e:
        logging.error(f"❌ Error making database accessible: {e}")
        return False

def create_database_summary():
    """Create a summary file for the frontend"""
    try:
        conn = sqlite3.connect("pokemon_cards.db")
        
        # Get basic stats
        total_cards = pd.read_sql_query("SELECT COUNT(*) as count FROM price_history", conn).iloc[0]['count']
        cards_with_prices = pd.read_sql_query(f"SELECT COUNT(*) as count FROM price_history WHERE '{price_column}' > 0", conn).iloc[0]['count']
        
        # Get latest price data
        latest_prices = pd.read_sql_query(f"""
            SELECT c.name, p."{price_column}" as price
            FROM card_info c
            JOIN price_history p ON c.id = p.id
            WHERE p."{price_column}" > 0
            ORDER BY p."{price_column}" DESC
            LIMIT 10
        """, conn)
        
        summary = {
            "last_updated": datetime.now().isoformat(),
            "total_cards": int(total_cards),
            "cards_with_prices": int(cards_with_prices),
            "coverage_percentage": round((cards_with_prices/total_cards)*100, 1),
            "price_column": price_column,
            "top_cards": latest_prices.to_dict('records')
        }
        
        # Save summary to multiple locations
        summary_locations = [
            "logs/database_summary.json",
            "../poke_frontend/database_summary.json",
            "../database_summary.json"
        ]
        
        for location in summary_locations:
            os.makedirs(os.path.dirname(location), exist_ok=True)
            with open(location, 'w') as f:
                json.dump(summary, f, indent=2)
        
        logging.info(f"✅ Database summary created: {summary}")
        conn.close()
        return True
        
    except Exception as e:
        logging.error(f"❌ Error creating database summary: {e}")
        return False

def main():
    """Main function to update prices with comprehensive logging."""
    start_time = time.time()
    
    # Setup logging
    setup_logging()
    
    logging.info("🚀 Starting ROBUST GitHub Actions Price Update")
    logging.info(f"📅 Date: {today}")
    logging.info(f"💰 Price column: {price_column}")
    logging.info(f"🔑 API Key: {'Configured' if api_key else 'Not configured'}")
    logging.info(f"🏠 Working directory: {os.getcwd()}")
    logging.info(f"🐍 Python version: {sys.version}")
    
    # Check if database exists
    if not os.path.exists("pokemon_cards.db"):
        logging.error("❌ Database file not found!")
        sys.exit(1)
    
    # Add new price column to database
    add_price_column_to_database(price_column)
    
    # Get existing card IDs from database
    existing_card_ids = get_existing_card_ids()
    logging.info(f"📊 Found {len(existing_card_ids)} existing cards in database")
    
    # Initialize progress tracking
    progress_data = {
        "start_time": datetime.now().isoformat(),
        "total_cards": len(existing_card_ids),
        "cards_processed": 0,
        "cards_updated": 0,
        "cards_no_price": 0,
        "cards_failed": 0,
        "current_page": 0,
        "status": "running"
    }
    
    # Get all cards from API using batch processing
    logging.info("🔄 Fetching all cards from API using batch processing...")
    cards_processed = 0
    cards_updated = 0
    cards_no_price = 0
    cards_failed = 0
    page = 1
    page_size = 250
    max_pages = 100  # Safety limit
    
    while page <= max_pages:
        logging.info(f"📄 Fetching page {page}/{max_pages}...")
        
        # Update progress
        progress_data["current_page"] = page
        progress_data["cards_processed"] = cards_processed
        progress_data["cards_updated"] = cards_updated
        progress_data["cards_no_price"] = cards_no_price
        progress_data["cards_failed"] = cards_failed
        create_progress_file(progress_data)
        
        # Add delay to avoid rate limiting
        delay = random.uniform(2.0, 3.0)
        logging.info(f"⏱️ Waiting {delay:.1f} seconds to avoid rate limiting...")
        time.sleep(delay)
        
        try:
            cards = Card.where(page=page, pageSize=page_size)
        except Exception as api_error:
            error_msg = str(api_error)
            if isinstance(api_error, bytes):
                try:
                    error_msg = api_error.decode('utf-8')
                except:
                    error_msg = "Unknown API error"
            
            logging.error(f"❌ API error on page {page}: {error_msg}")
            logging.info("🔄 Retrying in 15 seconds...")
            time.sleep(15)
            continue
        
        if not cards:
            logging.info("✅ No more cards found, ending pagination.")
            break
            
        # Process cards for this page
        page_cards_processed = 0
        page_cards_updated = 0
        
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
                                page_cards_updated += 1
                                logging.debug(f"✅ Updated: {safe_str(card.name)} - Price: ${market_price:.2f}")
                            else:
                                cards_no_price += 1
                                logging.debug(f"ℹ️ No price: {safe_str(card.name)}")
                        
                        cards_processed += 1
                        page_cards_processed += 1
                        
                    except (AttributeError, TypeError) as e:
                        logging.warning(f"⚠️ Error getting price for {safe_str(card.name)}: {e}")
                        update_card_price_in_database(card_id, None, price_column)
                        cards_no_price += 1
                        cards_processed += 1
                        page_cards_processed += 1
                else:
                    logging.debug(f"⏭️ Skipping {safe_str(card.name)} - not in database")
                    
            except Exception as e:
                logging.error(f"❌ Error processing card: {str(e)}")
                cards_failed += 1
                continue
        
        logging.info(f"📊 Page {page} completed: {page_cards_processed} processed, {page_cards_updated} updated")
        page += 1
    
    # Calculate final statistics
    end_time = time.time()
    duration = end_time - start_time
    coverage = (cards_updated/cards_processed)*100 if cards_processed > 0 else 0
    
    # Final statistics
    logging.info("=" * 60)
    logging.info("🎉 ROBUST Price Update Completed!")
    logging.info("=" * 60)
    logging.info(f"⏱️ Total duration: {duration/60:.1f} minutes")
    logging.info(f"📊 Total cards processed: {cards_processed:,}")
    logging.info(f"✅ Cards with prices: {cards_updated:,}")
    logging.info(f"ℹ️ Cards without prices: {cards_no_price:,}")
    logging.info(f"❌ Cards that failed: {cards_failed:,}")
    logging.info(f"📈 Coverage: {coverage:.1f}%")
    
    # Show final database statistics
    conn = sqlite3.connect("pokemon_cards.db")
    try:
        # Count cards with prices
        result = pd.read_sql_query(f"SELECT COUNT(*) as count FROM price_history WHERE '{price_column}' IS NOT NULL AND '{price_column}' > 0", conn)
        cards_with_prices = result['count'].iloc[0]
        
        # Get average price
        result = pd.read_sql_query(f"SELECT AVG('{price_column}') as avg_price FROM price_history WHERE '{price_column}' IS NOT NULL AND '{price_column}' > 0", conn)
        avg_price = result['avg_price'].iloc[0]
        
        logging.info(f"📊 Final Database Statistics:")
        logging.info(f"   Cards with prices: {cards_with_prices:,}")
        logging.info(f"   Average price: ${avg_price:.2f}")
        logging.info(f"   Coverage: {cards_with_prices/len(existing_card_ids)*100:.1f}%")
        
    finally:
        conn.close()
    
    # Make database accessible to frontend
    logging.info("🔗 Making database accessible to frontend...")
    if make_database_accessible():
        logging.info("✅ Database accessibility setup completed")
    else:
        logging.error("❌ Database accessibility setup failed")
    
    # Create database summary
    logging.info("📋 Creating database summary...")
    if create_database_summary():
        logging.info("✅ Database summary created")
    else:
        logging.error("❌ Database summary creation failed")
    
    # Update final progress
    progress_data.update({
        "end_time": datetime.now().isoformat(),
        "duration_minutes": round(duration/60, 1),
        "cards_processed": cards_processed,
        "cards_updated": cards_updated,
        "cards_no_price": cards_no_price,
        "cards_failed": cards_failed,
        "coverage_percentage": round(coverage, 1),
        "status": "completed"
    })
    create_progress_file(progress_data)
    
    # Exit with success
    logging.info("🎉 ROBUST GitHub Actions price update completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 