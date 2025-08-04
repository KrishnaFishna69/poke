#!/usr/bin/env python3
"""
Setup script for daily price updates using SQLite database
"""

import os
import sys
import subprocess
from pathlib import Path

def create_cron_job():
    """Create a cron job to run daily price updates"""
    
    # Get the absolute path to the script
    script_path = Path(__file__).parent / "pokeExp_database.py"
    script_path = script_path.absolute()
    
    # Get the Python interpreter path
    python_path = sys.executable
    
    # Create the cron command
    cron_command = f"0 9 * * * cd {script_path.parent} && {python_path} {script_path} >> price_update_cron.log 2>&1"
    
    print("🚀 Setting up daily price update cron job")
    print("=" * 50)
    print(f"Script path: {script_path}")
    print(f"Python path: {python_path}")
    print(f"Cron command: {cron_command}")
    print("\nTo add this to your crontab, run:")
    print(f"crontab -e")
    print("Then add this line:")
    print(f"{cron_command}")
    print("\nThis will run the price update every day at 9:00 AM")

def test_database_connection():
    """Test that the database is accessible"""
    
    try:
        import sqlite3
        conn = sqlite3.connect("pokemon_cards.db")
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("✅ Database connection successful")
        print(f"Tables found: {[table[0] for table in tables]}")
        
        # Check price_history table
        cursor.execute("PRAGMA table_info(price_history)")
        columns = [column[1] for column in cursor.fetchall()]
        price_columns = [col for col in columns if col.startswith('cardMarketPrice-')]
        
        print(f"Price columns: {len(price_columns)}")
        print(f"Latest price column: {max(price_columns) if price_columns else 'None'}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def show_database_stats():
    """Show current database statistics"""
    
    try:
        import sqlite3
        import pandas as pd
        
        conn = sqlite3.connect("pokemon_cards.db")
        
        # Get basic stats
        result = pd.read_sql_query("SELECT COUNT(*) as total_cards FROM price_history", conn)
        total_cards = result['total_cards'].iloc[0]
        
        # Get price columns
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(price_history)")
        columns = [column[1] for column in cursor.fetchall()]
        price_columns = [col for col in columns if col.startswith('cardMarketPrice-')]
        
        print(f"📊 Database Statistics:")
        print(f"Total cards: {total_cards}")
        print(f"Price history columns: {len(price_columns)}")
        
        if price_columns:
            # Get latest price data
            latest_col = max(price_columns)
            result = pd.read_sql_query(f"SELECT COUNT(*) as count FROM price_history WHERE '{latest_col}' IS NOT NULL", conn)
            cards_with_prices = result['count'].iloc[0]
            
            print(f"Latest price column: {latest_col}")
            print(f"Cards with latest prices: {cards_with_prices}")
            print(f"Coverage: {cards_with_prices/total_cards*100:.1f}%")
        
        conn.close()
        
    except Exception as e:
        print(f"Error getting database stats: {e}")

def main():
    """Main setup function"""
    
    print("🔧 Pokemon Cards Database Setup")
    print("=" * 50)
    
    # Test database
    if not test_database_connection():
        print("Please run convert_to_sqlite.py first to create the database")
        return
    
    # Show stats
    show_database_stats()
    
    print("\n" + "=" * 50)
    
    # Setup cron job
    create_cron_job()
    
    print("\n" + "=" * 50)
    print("📝 Manual Setup Instructions:")
    print("1. Test the script: python pokeExp_database.py")
    print("2. Add to crontab: crontab -e")
    print("3. Add the cron command shown above")
    print("4. Monitor logs: tail -f price_update_cron.log")

if __name__ == "__main__":
    main() 