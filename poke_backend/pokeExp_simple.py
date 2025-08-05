#!/usr/bin/env python3
"""
Simple Pokemon Card Price Update Script
Updates the SQLite database with today's prices
"""
print("Hello World")

from pokemontcgsdk import Card, RestClient
import pandas as pd
import sqlite3
import time
from datetime import datetime
import random
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure API
api_key = os.getenv('POKEMON_TCG_API_KEY', "85774024-7eef-4bc8-a628-3e3db902762e")
RestClient.configure(api_key)

# Get today's date
today = datetime.now().strftime("%m-%d-%y")
price_column = f"cardMarketPrice-{today}"

def safe_str(obj):
    """Safely convert object to string, handling bytes"""
    if obj is None:
        return None
    try:
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return str(obj)
    except Exception:
        return "Unknown error"

def add_price_column():
    """Add today's price column to database"""
    conn = sqlite3.connect("pokemon_cards.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute("PRAGMA table_info(price_history)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if price_column not in columns:
            cursor.execute(f"ALTER TABLE price_history ADD COLUMN '{price_column}' REAL")
            logging.info(f"✅ Added price column: {price_column}")
        else:
            logging.info(f"ℹ️ Price column {price_column} already exists")
    except Exception as e:
        logging.error(f"❌ Error adding column: {e}")
    finally:
        conn.close()

def get_existing_cards():
    """Get all card IDs from database"""
    conn = sqlite3.connect("pokemon_cards.db")
    try:
        df = pd.read_sql_query("SELECT id FROM price_history", conn)
        return set(df['id'].tolist())
    finally:
        conn.close()

def update_card_price(card_id, price):
    """Update price for a specific card"""
    conn = sqlite3.connect("pokemon_cards.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"UPDATE price_history SET '{price_column}' = ? WHERE id = ?", (price, card_id))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"❌ Error updating {card_id}: {e}")
        return False
    finally:
        conn.close()

def get_market_price(prices):
    """Extract market price from TCGPlayer prices"""
    if not prices:
        return None
    
    for price_type in ['holofoil', 'reverseHolofoil', 'normal', 'firstEditionHolofoil', 'firstEditionNormal']:
        price_obj = getattr(prices, price_type, None)
        if price_obj and hasattr(price_obj, 'market') and price_obj.market is not None:
            return float(price_obj.market)
    return None

def main():
    """Main update function"""
    start_time = time.time()
    
    logging.info("🚀 Starting Pokemon Card Price Update")
    logging.info(f"📅 Date: {today}")
    logging.info(f"💰 Price column: {price_column}")
    
    # Check database exists
    if not os.path.exists("pokemon_cards.db"):
        logging.error("❌ Database not found!")
        return 1
    
    # Add price column
    add_price_column()
    
    # Get existing cards
    existing_cards = get_existing_cards()
    logging.info(f"📊 Found {len(existing_cards)} cards in database")
    
    # Update prices
    cards_processed = 0
    cards_updated = 0
    cards_no_price = 0
    page = 1
    page_size = 250
    max_pages = 100  # Safety limit
    
    logging.info("🔄 Fetching cards from API...")
    
    while page <= max_pages:
        logging.info(f"📄 Processing page {page}...")
        
        # Rate limiting
        time.sleep(random.uniform(2.0, 3.0))
        
        try:
            cards = Card.where(page=page, pageSize=page_size)
        except Exception as e:
            error_msg = safe_str(e)
            logging.error(f"❌ API error on page {page}: {error_msg}")
            
            # If it's a 404, we've reached the end
            if "404" in error_msg or "Not Found" in error_msg:
                logging.info("✅ Reached end of cards (404 error)")
                break
            
            logging.info("🔄 Retrying in 15 seconds...")
            time.sleep(15)
            continue
        
        if not cards:
            logging.info("✅ No more cards found")
            break
        
        page_cards_updated = 0
        
        for card in cards:
            card_id = getattr(card, 'id', None)
            if not card_id or card_id not in existing_cards:
                continue
            
            try:
                # Get market price
                market_price = None
                if hasattr(card, 'tcgplayer') and hasattr(card.tcgplayer, 'prices'):
                    market_price = get_market_price(card.tcgplayer.prices)
                
                # Update database
                if update_card_price(card_id, market_price):
                    if market_price is not None:
                        cards_updated += 1
                        page_cards_updated += 1
                    else:
                        cards_no_price += 1
                    cards_processed += 1
                
            except Exception as e:
                card_name = safe_str(getattr(card, 'name', 'unknown'))
                logging.warning(f"⚠️ Error processing {card_name}: {safe_str(e)}")
                cards_no_price += 1
                cards_processed += 1
        
        logging.info(f"📊 Page {page}: {page_cards_updated} prices updated")
        page += 1
        
        # Progress update every 5 pages
        if page % 5 == 0:
            coverage = (cards_updated / cards_processed * 100) if cards_processed > 0 else 0
            logging.info(f"📈 Progress: {cards_processed:,} processed, {cards_updated:,} updated ({coverage:.1f}% coverage)")
    
    # Final statistics
    end_time = time.time()
    duration = end_time - start_time
    coverage = (cards_updated / cards_processed * 100) if cards_processed > 0 else 0
    
    logging.info("=" * 50)
    logging.info("🎉 Price Update Complete!")
    logging.info("=" * 50)
    logging.info(f"⏱️ Duration: {duration/60:.1f} minutes")
    logging.info(f"📊 Cards processed: {cards_processed:,}")
    logging.info(f"✅ Cards with prices: {cards_updated:,}")
    logging.info(f"ℹ️ Cards without prices: {cards_no_price:,}")
    logging.info(f"📈 Coverage: {coverage:.1f}%")
    
    # Database summary
    conn = sqlite3.connect("pokemon_cards.db")
    try:
        result = pd.read_sql_query(f"SELECT COUNT(*) as count FROM price_history WHERE '{price_column}' > 0", conn)
        total_with_prices = result['count'].iloc[0]
        
        result = pd.read_sql_query(f"SELECT AVG('{price_column}') as avg_price FROM price_history WHERE '{price_column}' > 0", conn)
        avg_price = result['avg_price'].iloc[0]
        
        logging.info(f"📊 Database Summary:")
        logging.info(f"   Total cards with prices: {total_with_prices:,}")
        logging.info(f"   Average price: ${avg_price:.2f}")
        logging.info(f"   Overall coverage: {total_with_prices/len(existing_cards)*100:.1f}%")
    finally:
        conn.close()
    
    logging.info("✅ Update completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 