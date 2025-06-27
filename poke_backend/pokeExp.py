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


# Configure the API client
RestClient.configure("98c0dc2b-311b-483e-9e86-f9ec67bbbdf0")
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
        df = pd.read_csv("pokemon_cards.csv")
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
    cards = Card.all()
    cards_processed = 0
    
    for card in cards:
        try:
            new_row = {}
            
            #Basic attributes that should always exist
            new_row["name"] = safe_get("name")
            new_row["id"] = safe_get("id")
            new_row["supertype"] = safe_get("supertype")
            
            #Complex attributes that might not exist
            new_row["subtypes"] = safe_get("subtypes")
            new_row["hp"] = safe_get("hp")
            new_row["types"] = safe_get("types")
            new_row["evolvesFrom"] = safe_get("evolvesFrom")
            new_row["evolvesTo"] = safe_get("evolvesTo")
            new_row["rules"] = safe_get("rules")
            new_row["abilities"] = safe_get("abilities")
            new_row["attacks"] = safe_get("attacks")
            new_row["weaknesses"] = safe_get("weaknesses")
            new_row["resistances"] = safe_get("resistances")
            new_row["retreatCost"] = safe_get("retreatCost")
            new_row["convertedRetreatCost"] = safe_get("convertedRetreatCost")
            new_row["set"] = safe_get("set")
            new_row["number"] = safe_get("number")
            new_row["artist"] = safe_get("artist")
            new_row["rarity"] = safe_get("rarity")
            new_row["flavorText"] = safe_get("flavorText")
            new_row["nationalPokedexNumber"] = safe_get("nationalPokedexNumber")
            new_row["legalities"] = safe_get("legalities")
            
            # Nested attributes that need special handling
            try:
                new_row["images"] = safe_str(card.images.large) if hasattr(card, 'images') else None
            except (AttributeError, TypeError):
                new_row["images"] = None
                
            try:
                market_price = get_market_price(card.tcgplayer.prices) if hasattr(card, 'tcgplayer') else None
                new_row[price_column] = market_price
                print(f"Card: {safe_str(card.name)}, Price: {market_price}")
            except (AttributeError, TypeError):
                new_row[price_column] = None
            
            # Only add cards that have pricing data
            if new_row[price_column] is not None:
                # Check if card already exists in DataFrame
                existing_card = df[df['id'] == new_row['id']]
                if not existing_card.empty:
                    # Update the price for existing card
                    df.loc[existing_card.index, price_column] = new_row[price_column]
                else:
                    # Add new card only if it has pricing data
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
                cards_processed += 1
                
                # Save every 100 cards in case of interruption
                if cards_processed % 100 == 0:
                    df.to_csv("pokemon_cards.csv", index=False)
                    print(f"Saved {cards_processed} cards with pricing data so far...")
            else:
                print(f"Skipping {safe_str(card.name)} - no pricing data available")
            
        except Exception as e:
            print(f"Error processing card: {str(e)}")
            continue

except Exception as e:
    print(f"An error occurred: {str(e)}")

# Final save
df.to_csv("pokemon_cards.csv", index=False)
print(f"\nData successfully saved to pokemon_cards.csv with {len(df)} cards")
print(f"Added new price column: {price_column}")


print("hey")