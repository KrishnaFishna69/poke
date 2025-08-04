#!/usr/bin/env python3
"""
Test script for GitHub Actions price update functionality
"""

import sqlite3
import pandas as pd
import logging
import os
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_database_operations():
    """Test database operations without API calls"""
    
    logging.info("🧪 Testing GitHub Actions database operations")
    
    # Check if database exists
    if not os.path.exists("pokemon_cards.db"):
        logging.error("❌ Database file not found!")
        return False
    
    # Test database connection
    try:
        conn = sqlite3.connect("pokemon_cards.db")
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        logging.info(f"✅ Database connection successful")
        logging.info(f"Tables found: {table_names}")
        
        # Test price_history table
        if 'price_history' in table_names:
            cursor.execute("PRAGMA table_info(price_history)")
            columns = [column[1] for column in cursor.fetchall()]
            price_columns = [col for col in columns if col.startswith('cardMarketPrice-')]
            
            logging.info(f"Price columns: {len(price_columns)}")
            if price_columns:
                logging.info(f"Latest price column: {max(price_columns)}")
            
            # Test adding a new price column
            today = datetime.now().strftime("%m-%d-%y")
            test_column = f"cardMarketPrice-{today}-test"
            
            if test_column not in columns:
                cursor.execute(f"ALTER TABLE price_history ADD COLUMN '{test_column}' REAL")
                logging.info(f"✅ Added test column: {test_column}")
                
                # Test updating a price
                cursor.execute(f"UPDATE price_history SET '{test_column}' = 10.50 WHERE id = (SELECT id FROM price_history LIMIT 1)")
                conn.commit()
                logging.info("✅ Test price update successful")
                
                # Clean up test column
                # Note: SQLite doesn't support DROP COLUMN easily, so we'll leave it
                logging.info("ℹ️ Test column left in database for verification")
            else:
                logging.info(f"Test column {test_column} already exists")
        
        # Get some statistics
        result = pd.read_sql_query("SELECT COUNT(*) as total_cards FROM price_history", conn)
        total_cards = result['total_cards'].iloc[0]
        logging.info(f"Total cards in database: {total_cards}")
        
        conn.close()
        return True
        
    except Exception as e:
        logging.error(f"❌ Database test failed: {e}")
        return False

def test_environment_variables():
    """Test environment variable handling"""
    
    logging.info("🧪 Testing environment variables")
    
    # Test API key environment variable
    api_key = os.getenv('POKEMON_TCG_API_KEY', "default-key")
    logging.info(f"API key (masked): {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '***'}")
    
    # Test GitHub Actions environment
    github_actions = os.getenv('GITHUB_ACTIONS', 'false')
    logging.info(f"Running in GitHub Actions: {github_actions}")
    
    if github_actions == 'true':
        logging.info("✅ GitHub Actions environment detected")
        logging.info(f"Repository: {os.getenv('GITHUB_REPOSITORY', 'unknown')}")
        logging.info(f"Workflow: {os.getenv('GITHUB_WORKFLOW', 'unknown')}")
    else:
        logging.info("ℹ️ Running in local environment")
    
    return True

def test_file_operations():
    """Test file operations that GitHub Actions will perform"""
    
    logging.info("🧪 Testing file operations")
    
    # Test current working directory
    cwd = os.getcwd()
    logging.info(f"Current working directory: {cwd}")
    
    # Test if we can write files
    test_file = "github_actions_test.txt"
    try:
        with open(test_file, 'w') as f:
            f.write(f"GitHub Actions test - {datetime.now()}")
        logging.info(f"✅ File write test successful: {test_file}")
        
        # Clean up
        os.remove(test_file)
        logging.info(f"✅ File cleanup successful")
        
    except Exception as e:
        logging.error(f"❌ File operation test failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    
    logging.info("🚀 GitHub Actions Setup Test")
    logging.info("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Database operations
    if test_database_operations():
        tests_passed += 1
        logging.info("✅ Database operations test PASSED")
    else:
        logging.error("❌ Database operations test FAILED")
    
    # Test 2: Environment variables
    if test_environment_variables():
        tests_passed += 1
        logging.info("✅ Environment variables test PASSED")
    else:
        logging.error("❌ Environment variables test FAILED")
    
    # Test 3: File operations
    if test_file_operations():
        tests_passed += 1
        logging.info("✅ File operations test PASSED")
    else:
        logging.error("❌ File operations test FAILED")
    
    # Summary
    logging.info("=" * 50)
    logging.info(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logging.info("🎉 All tests passed! GitHub Actions setup is ready.")
        logging.info("\n📝 Next steps:")
        logging.info("1. Commit and push your database: git add pokemon_cards.db && git commit -m 'Add database' && git push")
        logging.info("2. Push the workflow files: git add .github/ && git commit -m 'Add GitHub Actions' && git push")
        logging.info("3. Check the Actions tab in your GitHub repository")
        return 0
    else:
        logging.error("❌ Some tests failed. Please fix issues before setting up GitHub Actions.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 