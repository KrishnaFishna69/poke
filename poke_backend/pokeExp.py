from pokemontcgsdk import Card
from pokemontcgsdk import Set
from pokemontcgsdk import Type
from pokemontcgsdk import Supertype
from pokemontcgsdk import Subtype
from pokemontcgsdk import Rarity
from pokemontcgsdk import RestClient
import pandas as pd
import time
from datetime import datetime
import random

print("mmm\n")
# Configure the API client
RestClient.configure("85774024-7eef-4bc8-a628-3e3db902762e")
print("API client configured")
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

try:
    # Try to load existing CSV file
    try:
        df = pd.read_csv("pokemon_cards_clean.csv")
        print(f"Loaded existing CSV file with {len(df)} cards")
    except FileNotFoundError:
        # If file doesn't exist, create new DataFrame with all columns
        df = pd.DataFrame(columns=["name", "id", "supertype", "subtypes", "hp", "types",
                                 "evolvesFrom", "evolvesTo", "rules", "abilities", "attacks",
                                 "weaknesses", "resistances", "retreatCost", "convertedRetreatCost",
                                 "set", "number", "artist", "rarity", "flavorText",
                                 "nationalPokedexNumber", "legalities", "images"])
        print("Created new DataFrame")

    # Add new price column if it doesn't exist
    if price_column not in df.columns:
        df[price_column] = None

    # Get all cards
    print("\nFetching all cards...")
    cards_processed = 0
    print("Starting pagination loop...")  # Debug print
    page = 1
    page_size = 250  # Adjust as needed, API max is usually 250
    while True:
        print(f"Fetching page {page}...")  # Debug print
        # Add a small delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 1.5))
        
        # Try to fetch cards for this page
        try:
            cards = Card.where(page=page, pageSize=page_size)
        except Exception as api_error:
            print(f"API error on page {page}:")
            try:
                error_msg = api_error.decode('utf-8') if isinstance(api_error, bytes) else str(api_error)
                print(error_msg)
            except:
                print("Unknown API error occurred")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        
        if not cards:
            print("No more cards found, ending pagination.")  # Debug print
            break
            
        # Process cards for this page
        for card in cards:
            try:
                # Only update cards that already exist in the DataFrame
                card_id = safe_str(getattr(card, 'id', None))
                if card_id is None:
                    continue
                existing_card = df[df['id'] == card_id]
                if not existing_card.empty:
                    # Get the market price
                    try:
                        market_price = get_market_price(card.tcgplayer.prices) if hasattr(card, 'tcgplayer') else None
                        df.loc[existing_card.index, price_column] = market_price
                        print(f"Card: {safe_str(card.name)}, Price: {market_price}")
                    except (AttributeError, TypeError):
                        df.loc[existing_card.index, price_column] = None
                    cards_processed += 1
                    # Save every 100 cards in case of interruption
                    if cards_processed % 100 == 0:
                        df.to_csv("pokemon_cards_clean.csv", index=False)
                        print(f"Saved {cards_processed} cards with updated pricing data so far...")
                else:
                    print(f"Skipping {safe_str(card.name)} - card not in CSV")
            except Exception as e:
                print(f"Error processing card: {str(e)}")
                continue
        page += 1
except Exception as e:
    print("Error in outer try block:")
    if isinstance(e, bytes):
        print(e.decode('utf-8', errors='replace'))
    else:
        print(e)

# Final save
df.to_csv("pokemon_cards_clean.csv", index=False)
print(f"\nData successfully saved to pokemon_cards_clean.csv with {len(df)} cards")
print(f"Added new price column: {price_column}")


print("hey")