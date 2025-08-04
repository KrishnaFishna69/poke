import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_art_scores():
    """Merge art scores from art_scoring_complete.csv into pokemon_cards_clean.csv"""
    
    logging.info("Loading pokemon_cards_clean.csv...")
    clean_df = pd.read_csv("../pokemon_cards_clean.csv")
    logging.info(f"Loaded {len(clean_df)} cards from pokemon_cards_clean.csv")
    
    logging.info("Loading art_scoring_complete.csv...")
    art_df = pd.read_csv("art_scoring_complete.csv")
    logging.info(f"Loaded {len(art_df)} cards from art_scoring_complete.csv")
    
    # Extract only the art score columns from art_df
    art_scores = art_df[['id', 'pickscore_score', 'art_score_0_10']].copy()
    
    logging.info("Merging art scores...")
    # Merge on 'id' column
    merged_df = clean_df.merge(art_scores, on='id', how='left')
    
    # Check how many cards got art scores
    cards_with_scores = merged_df['art_score_0_10'].notna().sum()
    cards_without_scores = merged_df['art_score_0_10'].isna().sum()
    
    logging.info(f"Cards with art scores: {cards_with_scores}")
    logging.info(f"Cards without art scores: {cards_without_scores}")
    
    # Fill missing art scores with neutral value (5.0)
    if cards_without_scores > 0:
        logging.info(f"Filling {cards_without_scores} missing art scores with neutral value (5.0)")
        merged_df['art_score_0_10'] = merged_df['art_score_0_10'].fillna(5.0)
        merged_df['pickscore_score'] = merged_df['pickscore_score'].fillna(0.0)
    
    # Save the merged dataset
    output_file = "../pokemon_cards_clean_with_art.csv"
    merged_df.to_csv(output_file, index=False)
    logging.info(f"✅ Merged dataset saved to {output_file}")
    
    # Show some statistics
    logging.info(f"Final dataset: {len(merged_df)} cards")
    logging.info(f"Art score range: {merged_df['art_score_0_10'].min():.2f} - {merged_df['art_score_0_10'].max():.2f}")
    logging.info(f"Average art score: {merged_df['art_score_0_10'].mean():.2f}")
    
    return merged_df

if __name__ == "__main__":
    merge_art_scores() 