import pandas as pd
import requests
import json
import time
import logging
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comp_scoring.log'),
        logging.StreamHandler()
    ]
)

def create_competitive_token(row):
    """
    Create a competitive token string from card data if the card is Standard legal.
    """
    try:
        # Check if card is Standard legal - parse the legalities string
        legalities_str = str(row.get('legalities', ''))
        
        # Parse the legalities string format: "Legality(unlimited='Legal', expanded=None, standard=None)"
        if 'standard=' in legalities_str:
            # Extract the standard legality value
            import re
            standard_match = re.search(r"standard='([^']*)'", legalities_str)
            if standard_match:
                standard_legality = standard_match.group(1)
                if standard_legality != 'Legal':
                    return None
            else:
                # If standard is None or not found, card is not Standard legal
                return None
        else:
            # No standard legality info, assume not legal
            return None
        
        # Build competitive token string
        token_parts = []
        
        # Supertype
        if pd.notna(row.get('supertype')):
            token_parts.append(f"Supertype: {row['supertype']}")
        
        # Subtypes
        if pd.notna(row.get('subtypes')) and isinstance(row['subtypes'], list):
            subtypes_str = ', '.join(row['subtypes'])
            token_parts.append(f"Subtypes: {subtypes_str}")
        
        # HP
        if pd.notna(row.get('hp')):
            token_parts.append(f"HP: {row['hp']}")
        
        # Types
        if pd.notna(row.get('types')) and isinstance(row['types'], list):
            types_str = ', '.join(row['types'])
            token_parts.append(f"Types: {types_str}")
        
        # Rules
        if pd.notna(row.get('rules')) and isinstance(row['rules'], list):
            rules_str = '; '.join(row['rules'])
            token_parts.append(f"Rules: {rules_str}")
        
        # Abilities
        if pd.notna(row.get('abilities')) and isinstance(row['abilities'], list):
            abilities_str = '; '.join([f"{ability.get('name', '')}: {ability.get('text', '')}" 
                                     for ability in row['abilities']])
            token_parts.append(f"Abilities: {abilities_str}")
        
        # Attacks
        if pd.notna(row.get('attacks')) and isinstance(row['attacks'], list):
            attacks_str = '; '.join([f"{attack.get('name', '')}: {attack.get('text', '')} "
                                   f"(Damage: {attack.get('damage', 'N/A')}, "
                                   f"Cost: {', '.join(attack.get('convertedEnergyCost', []))})"
                                   for attack in row['attacks']])
            token_parts.append(f"Attacks: {attacks_str}")
        
        # Weaknesses
        if pd.notna(row.get('weaknesses')) and isinstance(row['weaknesses'], list):
            weaknesses_str = ', '.join([f"{weakness.get('type', '')} ({weakness.get('value', '')})"
                                      for weakness in row['weaknesses']])
            token_parts.append(f"Weaknesses: {weaknesses_str}")
        
        # Resistances
        if pd.notna(row.get('resistances')) and isinstance(row['resistances'], list):
            resistances_str = ', '.join([f"{resistance.get('type', '')} ({resistance.get('value', '')})"
                                       for resistance in row['resistances']])
            token_parts.append(f"Resistances: {resistances_str}")
        
        # Retreat Cost
        if pd.notna(row.get('retreatCost')):
            retreat_cost_str = ', '.join(row['retreatCost']) if isinstance(row['retreatCost'], list) else str(row['retreatCost'])
            token_parts.append(f"Retreat Cost: {retreat_cost_str}")
        
        # Converted Retreat Cost
        if pd.notna(row.get('convertedRetreatCost')):
            token_parts.append(f"Converted Retreat Cost: {row['convertedRetreatCost']}")
        
        return ' | '.join(token_parts) if token_parts else None
        
    except Exception as e:
        logging.error(f"Error creating competitive token for card {row.get('id', 'unknown')}: {str(e)}")
        return None

def query_ollama(prompt, model_name="qwen:latest", temperature=0.2):
    """
    Query Ollama API for competitive viability scoring.
    """
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()
        
    except Exception as e:
        logging.error(f"Error querying Ollama: {str(e)}")
        return None

def extract_score_from_response(response):
    """
    Extract numerical score from Ollama response with improved regex.
    """
    try:
        if not response:
            return None
        
        # Debug: log the actual response to see what we're getting
        logging.info(f"Raw LLM response: '{response}'")
        
        # Extract number 0-100 as a full response only
        match = re.search(r'^\s*(\d{1,3})(?:\.\d+)?\s*$', response.strip())
        if match:
            score = float(match.group(1))
            return max(0, min(100, score))  # Clamp between 0 and 100
        
        # Fallback: look for any number in the response
        numbers = re.findall(r'\b(\d{1,3})(?:\.\d+)?\b', response)
        if numbers:
            score = float(numbers[0])
            return max(0, min(100, score))
        
        # Additional fallback: look for numbers with decimal points
        decimal_match = re.search(r'(\d+\.\d+)', response)
        if decimal_match:
            score = float(decimal_match.group(1))
            return max(0, min(100, score))
        
        # Last resort: look for any number pattern
        any_number = re.search(r'(\d+(?:\.\d+)?)', response)
        if any_number:
            score = float(any_number.group(1))
            return max(0, min(100, score))
        
        logging.warning(f"Could not extract score from response: '{response}'")
        return None
    except Exception as e:
        logging.error(f"Error extracting score from response: {str(e)}")
        return None

def score_competitive_viability(competitive_token, temperature=0.2, retries=2):
    """
    Score competitive viability using Ollama with retry mechanism.
    """
    if not competitive_token:
        return None
    
    prompt = f"""
    You are an expert Pokémon TCG meta analyst AI. Your task is to assign a single real number score from 0 to 100 for each card, reflecting its competitive viability in the current meta. Scores should be consistent and precise across runs: a card's score represents its relative strength and impact on winning games today.

    Use the following guidelines:

    - 0 = completely unplayable or irrelevant in the current meta.
    - 100 = top-tier, dominant meta-defining card.
    - Scores between should reflect how useful and common the card is in competitive decks.

    Consider these examples:

    High-scoring meta Pokémon:

    - Dragapult EX (HP 170, Ability "Infiltrate" lets it move damage counters, fast 130 damage attack, used in aggressive decks for rapid pressure and disruption): 94.5

    - Gardevoir EX (HP 170, attacks that deal high damage and provide healing, synergy with Fairy energy, key control attacker in many decks): 89.7

    - Zacian V (HP 220, "Intrepid Sword" ability lets it draw cards and boost attack damage, 230 damage "Brave Blade" attack, staples in many decks): 92.0

    High-scoring meta Items and Supporters:

    - Choice Belt (Item, boosts damage done to opponent's Active Pokémon by 30, widely used for damage consistency): 82.3

    - Professor's Research (Supporter, discards your hand and draws 7 new cards, essential for refreshing resources): 96.2

    - Marnie (Supporter, shuffles opponent's hand into deck and draws cards yourself, great disruption and hand control): 88.5

    - Quick Ball (Item, allows searching Basic Pokémon quickly from the deck, key for fast setup): 85.0

    Low-scoring non-meta Pokémon:

    - Magikarp (HP 30, no attacks or abilities, purely basic Pokémon with no competitive use): 5.0

    - Caterpie (HP 40, weak attacks, no useful abilities, obsolete for competitive play): 8.0

    Low-scoring non-meta Items and Supporters:

    - Revive (Item, brings a Basic Pokémon from discard to hand but is slow and inefficient compared to modern recovery options): 10.0

    - Energy Retrieval (Item, retrieves up to 2 basic Energy cards from discard, but slow and outclassed by newer cards): 15.0

    - Professor Elm's Lecture (Supporter, draws 3 cards but is slow compared to other supporters and rarely played): 20.0

    When scoring, consider:

    - Card's stats (HP, damage output, abilities)
    - Playstyle and deck role (aggressive attacker, control, setup)
    - Popularity and presence in competitive decks
    - Synergies with other meta cards
    - Overall impact on win rates and strategies

    Provide a precise single real number score for each card reflecting its competitive viability:{competitive_token}
    """

    for attempt in range(retries + 1):
        response = query_ollama(prompt, temperature=temperature)
        score = extract_score_from_response(response)
        if score is not None:
            return score
        
        if attempt < retries:
            logging.warning(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(0.5)
    
    logging.warning("Failed to get valid score after all retries.")
    return None

def main():
    # Load dataset
    logging.info("Loading comp_scoring.csv...")
    df = pd.read_csv("comp_scoring.csv")
    logging.info(f"Loaded {len(df)} cards from comp_scoring.csv")
    
    # Create competitive tokens
    logging.info("Creating competitive tokens...")
    df['competitive_token'] = df.apply(create_competitive_token, axis=1)
    
    # Count legal cards
    legal_cards = df['competitive_token'].notna().sum()
    logging.info(f"Found {legal_cards} Standard legal cards out of {len(df)} total cards")
    
    # Process all legal cards
    legal_df = df[df['competitive_token'].notna()].copy()
    logging.info(f"Scoring {len(legal_df)} legal cards, 5 times each (temperature=0.2)...")
    
    # TEST MODE: Just test with first card to debug
    legal_df = legal_df.head(1).copy()
    logging.info(f"🧪 TEST MODE: Testing with just 1 card to debug")
    
    start_time = time.time()
    
    # Prepare columns for 5 runs
    for i in range(1, 6):
        df[f'CompViabilityScore_{i}'] = None
    
    # Score each card 5 times
    for idx, row in tqdm(legal_df.iterrows(), total=len(legal_df), desc="Scoring competitive viability"):
        for run in range(1, 6):
            score = score_competitive_viability(row['competitive_token'], temperature=0.2)
            df.at[idx, f'CompViabilityScore_{run}'] = score
            logging.info(f"Card {row.get('id', 'unknown')} run {run}: {score}")
            time.sleep(0.2)
    
    # Compute average
    score_cols = [f'CompViabilityScore_{i}' for i in range(1, 6)]
    df['CompViabilityScore_Avg'] = df[score_cols].astype(float).mean(axis=1)
    
    end_time = time.time()
    logging.info(f"✅ Completed competitive scoring in {end_time - start_time:.2f} seconds")
    
    # Save results
    df.to_csv("comp_scoring.csv", index=False)
    logging.info("Results saved to comp_scoring.csv")
    
    # Print summary statistics
    if 'CompViabilityScore_Avg' in df.columns:
        valid_scores = df['CompViabilityScore_Avg'].dropna()
        if len(valid_scores) > 0:
            logging.info(f"Competitive scoring summary:")
            logging.info(f"  Cards scored: {len(valid_scores)}")
            logging.info(f"  Average score: {valid_scores.mean():.2f}")
            logging.info(f"  Min score: {valid_scores.min():.2f}")
            logging.info(f"  Max score: {valid_scores.max():.2f}")
            
            # Show the first few scored cards
            scored_cards = df[df['CompViabilityScore_Avg'].notna()][['id', 'name', 'CompViabilityScore_Avg']].head(5)
            logging.info("First few scored cards:")
            for _, card in scored_cards.iterrows():
                logging.info(f"  {card['id']} - {card['name']}: {card['CompViabilityScore_Avg']}")
        else:
            logging.info("No valid scores found.")
    else:
        logging.info("CompViabilityScore_Avg column not created.")
    
    logging.info("Full experiment complete!")

if __name__ == "__main__":
    main()
