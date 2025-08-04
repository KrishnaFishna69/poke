#!/usr/bin/env python3
"""
Test script to verify GitHub Actions setup without running full update
"""

import os
import sqlite3
import json
import logging
from datetime import datetime

def test_setup():
    """Test the GitHub Actions setup"""
    
    print("🧪 Testing GitHub Actions Setup")
    print("=" * 50)
    
    # Test 1: Database exists and is accessible
    print("1. Testing database accessibility...")
    if os.path.exists("pokemon_cards.db"):
        print("   ✅ Database file exists")
        
        # Test database connection
        try:
            conn = sqlite3.connect("pokemon_cards.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_history")
            count = cursor.fetchone()[0]
            print(f"   ✅ Database connection successful: {count:,} cards")
            conn.close()
        except Exception as e:
            print(f"   ❌ Database connection failed: {e}")
            return False
    else:
        print("   ❌ Database file not found")
        return False
    
    # Test 2: Logs directory creation
    print("2. Testing logs directory creation...")
    try:
        os.makedirs("logs", exist_ok=True)
        print("   ✅ Logs directory created")
        
        # Test log file creation
        test_log = "logs/test_log.txt"
        with open(test_log, 'w') as f:
            f.write(f"Test log entry - {datetime.now()}")
        print("   ✅ Log file creation successful")
        
        # Clean up
        os.remove(test_log)
        print("   ✅ Log file cleanup successful")
        
    except Exception as e:
        print(f"   ❌ Logs directory test failed: {e}")
        return False
    
    # Test 3: Progress file creation
    print("3. Testing progress file creation...")
    try:
        progress_data = {
            "test": True,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("logs/test_progress.json", 'w') as f:
            json.dump(progress_data, f, indent=2)
        print("   ✅ Progress file creation successful")
        
        # Clean up
        os.remove("logs/test_progress.json")
        print("   ✅ Progress file cleanup successful")
        
    except Exception as e:
        print(f"   ❌ Progress file test failed: {e}")
        return False
    
    # Test 4: Database accessibility setup
    print("4. Testing database accessibility setup...")
    try:
        # Test frontend directory access
        frontend_dir = "../poke_frontend"
        if not os.path.exists(frontend_dir):
            os.makedirs(frontend_dir)
            print("   ✅ Frontend directory created")
        
        # Test database copy
        import shutil
        test_db_path = "../poke_frontend/test_db.db"
        shutil.copy2("pokemon_cards.db", test_db_path)
        print("   ✅ Database copy to frontend successful")
        
        # Clean up
        os.remove(test_db_path)
        print("   ✅ Database copy cleanup successful")
        
    except Exception as e:
        print(f"   ❌ Database accessibility test failed: {e}")
        return False
    
    # Test 5: Summary file creation
    print("5. Testing summary file creation...")
    try:
        summary_data = {
            "test": True,
            "total_cards": 18178,
            "last_updated": datetime.now().isoformat()
        }
        
        summary_locations = [
            "logs/test_summary.json",
            "../poke_frontend/test_summary.json"
        ]
        
        for location in summary_locations:
            os.makedirs(os.path.dirname(location), exist_ok=True)
            with open(location, 'w') as f:
                json.dump(summary_data, f, indent=2)
        
        print("   ✅ Summary file creation successful")
        
        # Clean up
        for location in summary_locations:
            if os.path.exists(location):
                os.remove(location)
        print("   ✅ Summary file cleanup successful")
        
    except Exception as e:
        print(f"   ❌ Summary file test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! GitHub Actions setup is ready.")
    print("\n📝 Next steps:")
    print("1. Commit and push: git add . && git commit -m 'Add robust GitHub Actions' && git push")
    print("2. Test manually: Go to Actions tab and run workflow manually")
    print("3. Monitor logs: Check the logs artifact after completion")
    print("4. Verify database: Check database_summary.json files")
    
    return True

if __name__ == "__main__":
    success = test_setup()
    if not success:
        print("\n❌ Some tests failed. Please fix issues before proceeding.")
        exit(1) 