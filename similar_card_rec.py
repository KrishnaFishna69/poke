import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import ast
import re

def parse_list_string(s):
    """Parse string representations of lists safely"""
    if pd.isna(s) or s == '':
        return []
    try:
        if isinstance(s, str):
            return ast.literal_eval(s)
        return s
    except:
        return []

def extract_attack_info(attacks_str):
    """Extract attack information from attacks string with more detailed stats"""
    if pd.isna(attacks_str) or attacks_str == '':
        return {'total_attacks': 0, 'avg_damage': 0, 'avg_cost': 0, 'max_damage': 0, 'min_damage': 0}
    
    try:
        attacks = parse_list_string(attacks_str)
        if not attacks:
            return {'total_attacks': 0, 'avg_damage': 0, 'avg_cost': 0, 'max_damage': 0, 'min_damage': 0}
        
        total_attacks = len(attacks)
        total_damage = 0
        total_cost = 0
        damage_values = []
        
        for attack in attacks:
            # Extract damage
            if 'damage' in attack:
                damage_str = str(attack['damage'])
                # Extract numeric damage
                damage_match = re.search(r'(\d+)', damage_str)
                if damage_match:
                    damage_val = int(damage_match.group(1))
                    total_damage += damage_val
                    damage_values.append(damage_val)
            
            # Extract cost
            if 'convertedEnergyCost' in attack:
                total_cost += attack['convertedEnergyCost']
        
        return {
            'total_attacks': total_attacks,
            'avg_damage': total_damage / total_attacks if total_attacks > 0 else 0,
            'avg_cost': total_cost / total_attacks if total_attacks > 0 else 0,
            'max_damage': max(damage_values) if damage_values else 0,
            'min_damage': min(damage_values) if damage_values else 0
        }
    except:
        return {'total_attacks': 0, 'avg_damage': 0, 'avg_cost': 0, 'max_damage': 0, 'min_damage': 0}

def build_feature_vector(row):
    """Build a feature vector for a card based purely on gameplay mechanics"""
    features = {}
    
    # Basic card stats
    features['hp'] = float(row['hp']) if pd.notna(row['hp']) else 0
    features['convertedRetreatCost'] = float(row['convertedRetreatCost']) if pd.notna(row['convertedRetreatCost']) else 0
    
    # Type encoding (one-hot encoding for all types)
    types = parse_list_string(row['types'])
    type_mapping = {
        'Grass': 0, 'Fire': 1, 'Water': 2, 'Lightning': 3, 'Psychic': 4,
        'Fighting': 5, 'Darkness': 6, 'Metal': 7, 'Fairy': 8, 'Dragon': 9,
        'Colorless': 10
    }
    
    for type_name in type_mapping.keys():
        features[f'type_{type_name}'] = 1 if type_name in types else 0
    
    # Subtype encoding (evolution stage and special types)
    subtypes = parse_list_string(row['subtypes'])
    features['is_basic'] = 1 if 'Basic' in subtypes else 0
    features['is_stage1'] = 1 if 'Stage 1' in subtypes else 0
    features['is_stage2'] = 1 if 'Stage 2' in subtypes else 0
    features['is_vmax'] = 1 if 'VMAX' in subtypes else 0
    features['is_v'] = 1 if 'V' in subtypes else 0
    features['is_gx'] = 1 if 'GX' in subtypes else 0
    features['is_ex'] = 1 if 'ex' in subtypes else 0
    features['is_legend'] = 1 if 'Legend' in subtypes else 0
    features['is_break'] = 1 if 'BREAK' in subtypes else 0
    features['is_prime'] = 1 if 'Prime' in subtypes else 0
    features['is_lv_x'] = 1 if 'LV.X' in subtypes else 0
    
    # Rarity encoding
    rarity = str(row['rarity']).lower() if pd.notna(row['rarity']) else ''
    features['rarity_common'] = 1 if 'common' in rarity else 0
    features['rarity_uncommon'] = 1 if 'uncommon' in rarity else 0
    features['rarity_rare'] = 1 if 'rare' in rarity else 0
    features['rarity_ultra_rare'] = 1 if 'ultra rare' in rarity or 'secret rare' in rarity else 0
    features['rarity_holo'] = 1 if 'holo' in rarity else 0
    
    # Attack features (mechanical properties)
    attack_info = extract_attack_info(row['attacks'])
    features['total_attacks'] = attack_info['total_attacks']
    features['avg_damage'] = attack_info['avg_damage']
    features['avg_cost'] = attack_info['avg_cost']
    features['max_damage'] = attack_info['max_damage']
    features['min_damage'] = attack_info['min_damage']
    
    # Abilities and special effects
    abilities = parse_list_string(row['abilities'])
    features['has_abilities'] = 1 if abilities else 0
    features['num_abilities'] = len(abilities)
    
    # Weaknesses and resistances
    weaknesses = parse_list_string(row['weaknesses'])
    resistances = parse_list_string(row['resistances'])
    features['num_weaknesses'] = len(weaknesses)
    features['num_resistances'] = len(resistances)
    
    # Evolution chain info
    features['evolves_from'] = 1 if pd.notna(row['evolvesFrom']) and row['evolvesFrom'] != '' else 0
    features['evolves_to'] = 1 if pd.notna(row['evolvesTo']) and row['evolvesTo'] != '' else 0
    
    # Rules and special conditions
    rules = str(row['rules']) if pd.notna(row['rules']) else ''
    features['has_special_rules'] = 1 if rules and rules != '' else 0
    
    # Flavor text (indicates if card has descriptive text)
    flavor_text = str(row['flavorText']) if pd.notna(row['flavorText']) else ''
    features['has_flavor_text'] = 1 if flavor_text and flavor_text != '' else 0
    
    # National Pokedex number (for evolutionary relationships)
    features['has_pokedex_number'] = 1 if pd.notna(row['nationalPokedexNumber']) else 0
    
    # Set information (era/format)
    set_info = str(row['set']) if pd.notna(row['set']) else ''
    features['is_modern'] = 1 if any(era in set_info.lower() for era in ['sword', 'shield', 'scarlet', 'violet', 'sun', 'moon']) else 0
    features['is_classic'] = 1 if any(era in set_info.lower() for era in ['base', 'jungle', 'fossil', 'team rocket']) else 0
    
    return features

def find_similar_cards(target_card_name, target_features, all_cards_data, top_n=5):
    """Find the most similar cards using cosine similarity, excluding duplicates"""
    # Filter out the target card and its duplicates
    filtered_cards = []
    filtered_features = []
    
    for i, (card_name, features) in enumerate(all_cards_data):
        # Skip if it's the same card name (duplicates)
        if card_name == target_card_name:
            continue
        filtered_cards.append(card_name)
        filtered_features.append(features)
    
    if not filtered_features:
        return []
    
    # Convert features to numpy array
    feature_matrix = np.array([list(features.values()) for features in filtered_features])
    
    # Add target features to the matrix for comparison
    target_feature_array = np.array([list(target_features.values())])
    full_matrix = np.vstack([target_feature_array, feature_matrix])
    
    # Normalize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(full_matrix)
    
    # Calculate cosine similarity between target and all other cards
    similarities = cosine_similarity([feature_matrix_scaled[0]], feature_matrix_scaled[1:])[0]
    
    # Get indices of top similar cards
    similar_indices = np.argsort(similarities)[::-1][:top_n]
    
    return [(filtered_cards[i], similarities[i]) for i in similar_indices]

def main():
    print("Loading Pokemon cards data...")
    
    # Read the CSV file
    df = pd.read_csv('pokemon_cards.csv', low_memory=False)
    
    print(f"Loaded {len(df)} cards")
    
    # Remove duplicates based on name and keep the one with highest popularity
    df_deduplicated = df.sort_values('popularity_votes', ascending=False).drop_duplicates(subset=['name'], keep='first')
    
    print(f"After removing duplicates: {len(df_deduplicated)} unique cards")
    
    # Sort by popularity votes to get top 5
    df_sorted = df_deduplicated.sort_values('popularity_votes', ascending=False)
    top_5_cards = df_sorted.head(5)
    
    print("\nTop 5 most popular cards:")
    for i, (_, card) in enumerate(top_5_cards.iterrows(), 1):
        print(f"{i}. {card['name']} (Votes: {card['popularity_votes']})")
    
    print("\nBuilding feature vectors for all cards...")
    
    # Build feature vectors for all cards
    all_cards_data = []
    
    for _, card in df_deduplicated.iterrows():
        features = build_feature_vector(card)
        all_cards_data.append((card['name'], features))
    
    print(f"Built feature vectors for {len(all_cards_data)} cards")
    
    # Find similar cards for each of the top 5
    print("\n" + "="*80)
    print("SIMILAR CARD RECOMMENDATIONS")
    print("="*80)
    
    for i, (_, card) in enumerate(top_5_cards.iterrows(), 1):
        print(f"\n{i}. {card['name']} (Votes: {card['popularity_votes']})")
        print("-" * 60)
        
        # Build feature vector for this card
        target_features = build_feature_vector(card)
        
        # Find similar cards
        similar_cards = find_similar_cards(card['name'], target_features, all_cards_data, top_n=5)
        
        if similar_cards:
            print("Most similar cards:")
            for j, (similar_name, similarity_score) in enumerate(similar_cards, 1):
                print(f"  {j}. {similar_name} (Similarity: {similarity_score:.3f})")
        else:
            print("No similar cards found.")

if __name__ == "__main__":
    main()
