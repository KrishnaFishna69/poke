import pandas as pd
import re

# Load data
df_pop = pd.read_csv("poke_votes.csv")  # Columns: Name, first_place_votes, top_six_votes, total_votes
df_cards = pd.read_csv("cards.csv", low_memory=False)     # Columns: name, id, supertype, etc.

# Normalize names
df_pop['Name'] = df_pop['Name'].str.lower().str.strip()
pokemon_names = set(df_pop['Name'])

def extract_pokemon_names(card_name):
    if pd.isna(card_name):
        return []
    
    name = str(card_name).lower()
    # Replace symbols like "-" or "&" with spaces
    name = re.sub(r'[-/&]', ' ', name)
    # Split into tokens
    tokens = re.findall(r'\b[a-z]+\b', name)
    
    # Extract all matching Pokémon names
    matched = [p for p in pokemon_names if p in name]
    return list(set(matched))  # Deduplicate if needed

# Apply to card dataset
df_cards['matched_pokemon'] = df_cards['name'].apply(extract_pokemon_names)

# Helper to compute total votes
def get_total_votes(pokemon_list):
    if not pokemon_list:
        return 0
    return df_pop[df_pop['Name'].isin(pokemon_list)]['total_votes'].sum()

# Add popularity score to cards
df_cards['popularity_votes'] = df_cards['matched_pokemon'].apply(get_total_votes)

# Optional: average votes instead of total
def get_avg_votes(pokemon_list):
    if not pokemon_list:
        return 0
    return df_pop[df_pop['Name'].isin(pokemon_list)]['total_votes'].mean()

df_cards['avg_votes'] = df_cards['matched_pokemon'].apply(get_avg_votes)

# Print some statistics
total_cards = len(df_cards)
matched_cards = len(df_cards[df_cards['popularity_votes'] > 0])
print(f"Total cards: {total_cards}")
print(f"Cards with popularity votes: {matched_cards} ({matched_cards/total_cards*100:.1f}%)")

# Show some examples
print("\nExample matches:")
sample_matches = df_cards[df_cards['popularity_votes'] > 0].head(10)
for _, row in sample_matches.iterrows():
    print(f"  '{row['name']}' → {row['matched_pokemon']} (votes: {row['popularity_votes']})")

# Export
df_cards.to_csv("cards_with_popularity.csv", index=False)
print(f"\nResults saved to cards_with_popularity.csv")
