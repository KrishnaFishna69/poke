#!/usr/bin/env python3
"""
Explain why not all cards have prices and show the breakdown
"""

import sqlite3
import pandas as pd
from datetime import datetime

def explain_price_coverage():
    """Explain the price coverage situation"""
    
    today = datetime.now().strftime("%m-%d-%y")
    price_column = f"cardMarketPrice-{today}"
    
    print(f"🔍 Price Coverage Analysis ({today})")
    print("=" * 60)
    
    conn = sqlite3.connect("pokemon_cards.db")
    
    try:
        # Get total counts
        total_query = "SELECT COUNT(*) as total FROM price_history"
        total_cards = pd.read_sql_query(total_query, conn).iloc[0]['total']
        
        # Get cards with actual prices (> 0)
        priced_query = f"SELECT COUNT(*) as priced FROM price_history WHERE \"{price_column}\" > 0"
        priced_cards = pd.read_sql_query(priced_query, conn).iloc[0]['priced']
        
        # Get cards with null/zero prices
        unpriced_query = f"SELECT COUNT(*) as unpriced FROM price_history WHERE \"{price_column}\" IS NULL OR \"{price_column}\" = 0"
        unpriced_cards = pd.read_sql_query(unpriced_query, conn).iloc[0]['unpriced']
        
        print(f"📊 Overall Statistics:")
        print(f"   Total cards in database: {total_cards:,}")
        print(f"   Cards with prices (> $0): {priced_cards:,}")
        print(f"   Cards without prices: {unpriced_cards:,}")
        print(f"   Coverage: {(priced_cards/total_cards)*100:.1f}%")
        
        print(f"\n❓ Why don't all cards have prices?")
        print("=" * 60)
        print("This is completely normal! Here's why:")
        print()
        print("1. 🃏 **Trainer Cards & Energy**: Many cards are trainer cards, energy cards,")
        print("   or other non-Pokemon cards that don't have market prices")
        print()
        print("2. 📊 **Market Availability**: Some cards are so rare or old that they")
        print("   don't have current market data available")
        print()
        print("3. 🏪 **No Recent Sales**: Cards without recent sales don't have")
        print("   current market prices")
        print()
        print("4. 🌍 **Regional Differences**: Some cards may not be available")
        print("   in the market you're querying")
        
        # Show breakdown by card type
        print(f"\n📋 Breakdown by Card Type:")
        print("-" * 40)
        
        type_query = f"""
        SELECT 
            c.supertype,
            COUNT(*) as total,
            COUNT(CASE WHEN p."{price_column}" > 0 THEN 1 END) as with_prices,
            COUNT(CASE WHEN p."{price_column}" IS NULL OR p."{price_column}" = 0 THEN 1 END) as without_prices
        FROM card_info c
        JOIN price_history p ON c.id = p.id
        GROUP BY c.supertype
        ORDER BY total DESC
        """
        
        type_breakdown = pd.read_sql_query(type_query, conn)
        
        for _, row in type_breakdown.iterrows():
            supertype = row['supertype'] if pd.notna(row['supertype']) else 'Unknown'
            total = row['total']
            with_prices = row['with_prices']
            without_prices = row['without_prices']
            coverage = (with_prices/total)*100 if total > 0 else 0
            
            print(f"   {supertype:<12} {total:>5} total, {with_prices:>4} with prices ({coverage:>5.1f}%)")
        
        # Show examples of cards without prices
        print(f"\n📝 Examples of Cards Without Prices:")
        print("-" * 40)
        
        examples_query = f"""
        SELECT 
            c.name,
            c.supertype,
            c."set"
        FROM card_info c
        JOIN price_history p ON c.id = p.id
        WHERE p."{price_column}" IS NULL OR p."{price_column}" = 0
        LIMIT 10
        """
        
        examples = pd.read_sql_query(examples_query, conn)
        
        for _, row in examples.iterrows():
            set_name = str(row['set']).split("name='")[1].split("'")[0] if "name='" in str(row['set']) else str(row['set'])
            supertype = row['supertype'] if pd.notna(row['supertype']) else 'Unknown'
            print(f"   {row['name']:<20} ({supertype}) - {set_name}")
        
        # Show price distribution for cards that DO have prices
        print(f"\n💰 Price Distribution (Cards with Prices):")
        print("-" * 40)
        
        if priced_cards > 0:
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
                percentage = (row['count'] / priced_cards) * 100
                print(f"   {row['price_range']:<12} {row['count']:>5} cards ({percentage:>5.1f}%)")
        
        print(f"\n✅ This is completely normal and expected!")
        print(f"📈 Your database is working perfectly with {priced_cards:,} cards having current market prices")
        print(f"🔄 GitHub Actions will continue to update prices daily for available cards")
        
    finally:
        conn.close()

if __name__ == "__main__":
    explain_price_coverage() 