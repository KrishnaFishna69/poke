#!/usr/bin/env python3
"""
Show today's prices summary from the database
"""

import sqlite3
import pandas as pd
from datetime import datetime

def show_today_prices():
    """Show a summary of today's prices"""
    
    today = datetime.now().strftime("%m-%d-%y")
    price_column = f"cardMarketPrice-{today}"
    
    print(f"📊 Today's Prices Summary ({today})")
    print("=" * 60)
    
    conn = sqlite3.connect("pokemon_cards.db")
    
    try:
        # Check if today's price column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(price_history)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if price_column not in columns:
            print(f"❌ No price data found for {today}")
            return
        
        # Get statistics
        stats_query = f"""
        SELECT 
            COUNT(*) as total_cards,
            COUNT("{price_column}") as cards_with_prices,
            AVG("{price_column}") as avg_price,
            MIN("{price_column}") as min_price,
            MAX("{price_column}") as max_price
        FROM price_history 
        WHERE "{price_column}" IS NOT NULL
        """
        
        stats = pd.read_sql_query(stats_query, conn)
        
        print(f"📈 Price Statistics:")
        print(f"   Total cards: {stats['total_cards'].iloc[0]:,}")
        print(f"   Cards with prices: {stats['cards_with_prices'].iloc[0]:,}")
        print(f"   Average price: ${stats['avg_price'].iloc[0]:.2f}")
        print(f"   Price range: ${stats['min_price'].iloc[0]:.2f} - ${stats['max_price'].iloc[0]:.2f}")
        
        # Top 10 most expensive cards
        print(f"\n🏆 Top 10 Most Expensive Cards:")
        print("-" * 40)
        
        top_query = f"""
        SELECT 
            c.name,
            c."set",
            p."{price_column}" as price
        FROM card_info c
        JOIN price_history p ON c.id = p.id
        WHERE p."{price_column}" > 0
        ORDER BY p."{price_column}" DESC
        LIMIT 10
        """
        
        top_cards = pd.read_sql_query(top_query, conn)
        
        for i, (_, row) in enumerate(top_cards.iterrows(), 1):
            set_name = str(row['set']).split("name='")[1].split("'")[0] if "name='" in str(row['set']) else str(row['set'])
            print(f"{i:2d}. {row['name']:<25} ${row['price']:>8.2f} ({set_name})")
        
        # Popular cards (Charizard, Pikachu, etc.)
        print(f"\n⭐ Popular Cards:")
        print("-" * 40)
        
        popular_query = f"""
        SELECT 
            c.name,
            p."{price_column}" as price
        FROM card_info c
        JOIN price_history p ON c.id = p.id
        WHERE (c.name LIKE '%Charizard%' OR c.name LIKE '%Pikachu%' OR c.name LIKE '%Mewtwo%')
        AND p."{price_column}" > 0
        ORDER BY p."{price_column}" DESC
        LIMIT 8
        """
        
        popular_cards = pd.read_sql_query(popular_query, conn)
        
        for _, row in popular_cards.iterrows():
            print(f"   {row['name']:<20} ${row['price']:>8.2f}")
        
        # Price distribution
        print(f"\n📊 Price Distribution:")
        print("-" * 40)
        
        distribution_query = f"""
        SELECT 
            CASE 
                WHEN "{price_column}" < 1 THEN 'Under $1'
                WHEN "{price_column}" < 5 THEN '$1-$5'
                WHEN "{price_column}" < 10 THEN '$5-$10'
                WHEN "{price_column}" < 25 THEN '$10-$25'
                WHEN "{price_column}" < 50 THEN '$25-$50'
                WHEN "{price_column}" < 100 THEN '$50-$100'
                WHEN "{price_column}" < 500 THEN '$100-$500'
                ELSE 'Over $500'
            END as price_range,
            COUNT(*) as count
        FROM price_history 
        WHERE "{price_column}" > 0
        GROUP BY price_range
        ORDER BY 
            CASE price_range
                WHEN 'Under $1' THEN 1
                WHEN '$1-$5' THEN 2
                WHEN '$5-$10' THEN 3
                WHEN '$10-$25' THEN 4
                WHEN '$25-$50' THEN 5
                WHEN '$50-$100' THEN 6
                WHEN '$100-$500' THEN 7
                ELSE 8
            END
        """
        
        distribution = pd.read_sql_query(distribution_query, conn)
        
        for _, row in distribution.iterrows():
            percentage = (row['count'] / stats['cards_with_prices'].iloc[0]) * 100
            print(f"   {row['price_range']:<12} {row['count']:>5} cards ({percentage:>5.1f}%)")
        
        print(f"\n✅ Database successfully updated with today's prices!")
        print(f"📅 Next update: Tomorrow at 12:00 PM UTC via GitHub Actions")
        
    finally:
        conn.close()

if __name__ == "__main__":
    show_today_prices() 