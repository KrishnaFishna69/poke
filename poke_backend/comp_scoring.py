#!/usr/bin/env python3
"""
Neural Network-Based Pokemon Card Competitive Viability Scoring
Uses card features to predict competitive viability scores
"""
print("lol")
import pandas as pd
import numpy as np
import sqlite3
import json
import logging
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neural_comp_scoring.log'),
        logging.StreamHandler()
    ]
)

class PokemonCardNeuralScorer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.numerical_features = []
        self.categorical_features = []
        
    def parse_attacks(self, attacks_str):
        """Parse attack information and extract numerical features"""
        if pd.isna(attacks_str) or not attacks_str:
            return {
                'attack_count': 0,
                'total_damage': 0,
                'avg_damage': 0,
                'max_damage': 0,
                'total_energy_cost': 0,
                'avg_energy_cost': 0,
                'has_special_effects': 0,
                'damage_per_energy': 0,
                'has_high_damage': 0,
                'has_low_cost': 0
            }
        
        try:
            if isinstance(attacks_str, str):
                # Parse string format
                attacks = json.loads(attacks_str.replace("'", '"'))
            elif isinstance(attacks_str, list):
                attacks = attacks_str
            else:
                return self.parse_attacks(None)
            
            attack_count = len(attacks)
            damages = []
            energy_costs = []
            has_special_effects = 0
            damage_per_energy_ratios = []
            
            for attack in attacks:
                # Extract damage
                damage = attack.get('damage', '0')
                if isinstance(damage, str):
                    # Convert damage strings to numbers
                    if damage == 'N/A' or damage == '':
                        damage = 0
                    else:
                        damage = int(damage) if damage.isdigit() else 0
                damages.append(damage)
                
                # Extract energy cost
                energy_cost = len(attack.get('convertedEnergyCost', []))
                energy_costs.append(energy_cost)
                
                # Calculate damage per energy ratio
                if energy_cost > 0:
                    damage_per_energy_ratios.append(damage / energy_cost)
                else:
                    damage_per_energy_ratios.append(0)
                
                # Check for special effects
                text = attack.get('text', '').lower()
                if any(keyword in text for keyword in ['draw', 'search', 'heal', 'prevent', 'switch', 'evolve', 'paralyze', 'poison', 'burn']):
                    has_special_effects = 1
            
            # Calculate additional features
            has_high_damage = 1 if max(damages) >= 120 else 0
            has_low_cost = 1 if min(energy_costs) <= 1 else 0
            avg_damage_per_energy = np.mean(damage_per_energy_ratios) if damage_per_energy_ratios else 0
            
            return {
                'attack_count': attack_count,
                'total_damage': sum(damages),
                'avg_damage': np.mean(damages) if damages else 0,
                'max_damage': max(damages) if damages else 0,
                'total_energy_cost': sum(energy_costs),
                'avg_energy_cost': np.mean(energy_costs) if energy_costs else 0,
                'has_special_effects': has_special_effects,
                'damage_per_energy': avg_damage_per_energy,
                'has_high_damage': has_high_damage,
                'has_low_cost': has_low_cost
            }
        except Exception as e:
            logging.warning(f"Error parsing attacks: {e}")
            return self.parse_attacks(None)
    
    def parse_weaknesses_resistances(self, weaknesses_str, resistances_str):
        """Parse weaknesses and resistances"""
        def parse_type_modifiers(modifiers_str):
            if pd.isna(modifiers_str) or not modifiers_str:
                return {'count': 0, 'total_modifier': 0, 'avg_modifier': 0, 'has_weakness': 0, 'has_resistance': 0}
            
            try:
                if isinstance(modifiers_str, str):
                    modifiers = json.loads(modifiers_str.replace("'", '"'))
                elif isinstance(modifiers_str, list):
                    modifiers = modifiers_str
                else:
                    return {'count': 0, 'total_modifier': 0, 'avg_modifier': 0, 'has_weakness': 0, 'has_resistance': 0}
                
                modifiers_list = []
                for modifier in modifiers:
                    value = modifier.get('value', '0')
                    if isinstance(value, str):
                        if value == 'N/A' or value == '':
                            value = 0
                        else:
                            # Extract number from strings like "×2", "+30", etc.
                            value = re.findall(r'[\d.]+', value)
                            value = float(value[0]) if value else 0
                    modifiers_list.append(value)
                
                return {
                    'count': len(modifiers_list),
                    'total_modifier': sum(modifiers_list),
                    'avg_modifier': np.mean(modifiers_list) if modifiers_list else 0,
                    'has_weakness': 1 if any(v > 1 for v in modifiers_list) else 0,
                    'has_resistance': 1 if any(v < 1 and v > 0 for v in modifiers_list) else 0
                }
            except Exception as e:
                logging.warning(f"Error parsing modifiers: {e}")
                return {'count': 0, 'total_modifier': 0, 'avg_modifier': 0, 'has_weakness': 0, 'has_resistance': 0}
        
        weaknesses = parse_type_modifiers(weaknesses_str)
        resistances = parse_type_modifiers(resistances_str)
        
        return {
            'weakness_count': weaknesses['count'],
            'weakness_avg': weaknesses['avg_modifier'],
            'resistance_count': resistances['count'],
            'resistance_avg': resistances['avg_modifier'],
            'has_weakness': weaknesses['has_weakness'],
            'has_resistance': resistances['has_resistance']
        }
    
    def extract_features(self, row):
        """Extract all features from a card row"""
        features = {}
        
        # Basic numerical features
        features['hp'] = float(row.get('hp', 0)) if pd.notna(row.get('hp')) else 0
        features['convertedRetreatCost'] = float(row.get('convertedRetreatCost', 0)) if pd.notna(row.get('convertedRetreatCost')) else 0
        
        # Categorical features
        features['supertype'] = str(row.get('supertype', 'Unknown'))
        
        # Subtypes (count and presence of key subtypes)
        subtypes = row.get('subtypes', [])
        if isinstance(subtypes, str):
            try:
                subtypes = json.loads(subtypes.replace("'", '"'))
            except:
                subtypes = []
        features['subtype_count'] = len(subtypes) if isinstance(subtypes, list) else 0
        features['is_basic'] = 1 if isinstance(subtypes, list) and 'Basic' in subtypes else 0
        features['is_stage1'] = 1 if isinstance(subtypes, list) and 'Stage 1' in subtypes else 0
        features['is_stage2'] = 1 if isinstance(subtypes, list) and 'Stage 2' in subtypes else 0
        features['is_vmax'] = 1 if isinstance(subtypes, list) and 'VMAX' in subtypes else 0
        features['is_v'] = 1 if isinstance(subtypes, list) and 'V' in subtypes else 0
        features['is_gx'] = 1 if isinstance(subtypes, list) and 'GX' in subtypes else 0
        features['is_ex'] = 1 if isinstance(subtypes, list) and 'EX' in subtypes else 0
        features['is_legendary'] = 1 if isinstance(subtypes, list) and 'Legendary' in subtypes else 0
        features['is_ultra_beast'] = 1 if isinstance(subtypes, list) and 'Ultra Beast' in subtypes else 0
        
        # Types (count and presence of key types)
        types = row.get('types', [])
        if isinstance(types, str):
            try:
                types = json.loads(types.replace("'", '"'))
            except:
                types = []
        features['type_count'] = len(types) if isinstance(types, list) else 0
        features['is_colorless'] = 1 if isinstance(types, list) and 'Colorless' in types else 0
        features['is_psychic'] = 1 if isinstance(types, list) and 'Psychic' in types else 0
        features['is_fighting'] = 1 if isinstance(types, list) and 'Fighting' in types else 0
        features['is_fire'] = 1 if isinstance(types, list) and 'Fire' in types else 0
        features['is_water'] = 1 if isinstance(types, list) and 'Water' in types else 0
        features['is_grass'] = 1 if isinstance(types, list) and 'Grass' in types else 0
        features['is_lightning'] = 1 if isinstance(types, list) and 'Lightning' in types else 0
        features['is_darkness'] = 1 if isinstance(types, list) and 'Darkness' in types else 0
        features['is_metal'] = 1 if isinstance(types, list) and 'Metal' in types else 0
        features['is_fairy'] = 1 if isinstance(types, list) and 'Fairy' in types else 0
        features['is_dragon'] = 1 if isinstance(types, list) and 'Dragon' in types else 0
        
        # Parse attacks
        attack_features = self.parse_attacks(row.get('attacks'))
        features.update(attack_features)
        
        # Parse weaknesses and resistances
        modifier_features = self.parse_weaknesses_resistances(
            row.get('weaknesses'), 
            row.get('resistances')
        )
        features.update(modifier_features)
        
        # Abilities (count and presence)
        abilities = row.get('abilities', [])
        if isinstance(abilities, str):
            try:
                abilities = json.loads(abilities.replace("'", '"'))
            except:
                abilities = []
        features['ability_count'] = len(abilities) if isinstance(abilities, list) else 0
        features['has_abilities'] = 1 if features['ability_count'] > 0 else 0
        
        # Rules text analysis
        rules = row.get('rules', [])
        if isinstance(rules, str):
            try:
                rules = json.loads(rules.replace("'", '"'))
            except:
                rules = []
        rules_text = ' '.join(rules) if isinstance(rules, list) else str(rules)
        features['rules_length'] = len(rules_text)
        features['has_complex_rules'] = 1 if len(rules_text) > 100 else 0
        
        # Set information (newer sets tend to be more competitive)
        set_name = str(row.get('set', {}).get('name', '')) if isinstance(row.get('set'), dict) else str(row.get('set', ''))
        features['is_modern_set'] = 1 if any(year in set_name for year in ['2023', '2024', '2025', 'Scarlet', 'Violet', 'Paldea']) else 0
        features['is_swsh_set'] = 1 if 'Sword' in set_name or 'Shield' in set_name else 0
        
        return features
    
    def prepare_data(self, df):
        """Prepare data for neural network training"""
        logging.info("🔄 Extracting features from cards...")
        
        all_features = []
        for idx, row in df.iterrows():
            try:
                features = self.extract_features(row)
                all_features.append(features)
            except Exception as e:
                logging.warning(f"Error extracting features for card {row.get('name', 'unknown')}: {e}")
                continue
        
        features_df = pd.DataFrame(all_features)
        
        # Separate numerical and categorical features
        self.numerical_features = [
            'hp', 'convertedRetreatCost', 'subtype_count', 'type_count',
            'attack_count', 'total_damage', 'avg_damage', 'max_damage',
            'total_energy_cost', 'avg_energy_cost', 'weakness_count',
            'weakness_avg', 'resistance_count', 'resistance_avg', 'ability_count',
            'rules_length', 'damage_per_energy'
        ]
        
        self.categorical_features = [
            'supertype', 'is_basic', 'is_stage1', 'is_stage2', 'is_vmax',
            'is_v', 'is_gx', 'is_ex', 'is_legendary', 'is_ultra_beast',
            'is_colorless', 'is_psychic', 'is_fighting', 'is_fire', 'is_water',
            'is_grass', 'is_lightning', 'is_darkness', 'is_metal', 'is_fairy',
            'is_dragon', 'has_special_effects', 'has_abilities', 'has_complex_rules',
            'has_high_damage', 'has_low_cost', 'has_weakness', 'has_resistance',
            'is_modern_set', 'is_swsh_set'
        ]
        
        # Encode categorical features
        for feature in self.categorical_features:
            if feature in features_df.columns:
                le = LabelEncoder()
                features_df[feature] = le.fit_transform(features_df[feature].astype(str))
                self.label_encoders[feature] = le
        
        # Scale numerical features
        numerical_data = features_df[self.numerical_features].fillna(0)
        scaled_numerical = self.scaler.fit_transform(numerical_data)
        features_df[self.numerical_features] = scaled_numerical
        
        self.feature_columns = self.numerical_features + self.categorical_features
        
        logging.info(f"✅ Extracted {len(features_df)} cards with {len(self.feature_columns)} features")
        return features_df
    
    def create_model(self, input_dim):
        """Create neural network model"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid')  # Output competitive score 0-1
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def generate_training_labels(self, df):
        """Generate competitive viability labels based on card features"""
        logging.info("🏷️ Generating competitive viability labels...")
        
        labels = []
        for idx, row in df.iterrows():
            score = 0.0
            
            # Base score from HP (0-20%)
            hp = float(row.get('hp', 0)) if pd.notna(row.get('hp')) else 0
            if hp > 0:
                hp_score = min(hp / 300.0, 0.2)
                score += hp_score
            
            # Attack damage contribution (0-25%)
            attacks = self.parse_attacks(row.get('attacks'))
            if attacks['max_damage'] > 0:
                damage_score = min(attacks['max_damage'] / 300.0, 0.25)
                score += damage_score
            
            # Damage per energy efficiency (0-15%)
            if attacks['damage_per_energy'] > 0:
                efficiency_score = min(attacks['damage_per_energy'] / 50.0, 0.15)
                score += efficiency_score
            
            # Ability bonus (0-15%)
            abilities = row.get('abilities', [])
            if isinstance(abilities, list) and len(abilities) > 0:
                score += 0.15
            
            # Type advantages (0-10%)
            types = row.get('types', [])
            if isinstance(types, list):
                if len(types) > 1:  # Multi-type cards
                    score += 0.05
                if 'Colorless' in types:  # Colorless is flexible
                    score += 0.05
            
            # Stage evolution bonus (0-15%)
            subtypes = row.get('subtypes', [])
            if isinstance(subtypes, list):
                if 'VMAX' in subtypes:
                    score += 0.15
                elif 'V' in subtypes:
                    score += 0.12
                elif 'GX' in subtypes:
                    score += 0.10
                elif 'EX' in subtypes:
                    score += 0.08
                elif 'Legendary' in subtypes:
                    score += 0.05
            
            # Special effects bonus (0-10%)
            if attacks['has_special_effects']:
                score += 0.10
            
            # Modern set bonus (0-5%)
            set_name = str(row.get('set', {}).get('name', '')) if isinstance(row.get('set'), dict) else str(row.get('set', ''))
            if any(year in set_name for year in ['2023', '2024', '2025', 'Scarlet', 'Violet', 'Paldea']):
                score += 0.05
            
            # Add some randomness to create variation
            import random
            random.seed(hash(row.get('name', '')) % 1000)  # Deterministic randomness
            score += random.uniform(-0.05, 0.05)
            
            # Normalize to 0-1 range
            score = max(0.0, min(score, 1.0))
            labels.append(score)
        
        logging.info(f"✅ Generated labels with average score: {np.mean(labels):.3f}")
        logging.info(f"   Score range: {min(labels):.3f} - {max(labels):.3f}")
        logging.info(f"   Score std: {np.std(labels):.3f}")
        return np.array(labels)
    
    def train_model(self, features_df, labels, test_size=0.2):
        """Train the neural network model"""
        logging.info("🧠 Training neural network model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df[self.feature_columns], labels, 
            test_size=test_size, random_state=42
        )
        
        # Create and train model
        self.model = self.create_model(len(self.feature_columns))
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logging.info(f"✅ Model trained successfully!")
        logging.info(f"   MSE: {mse:.4f}")
        logging.info(f"   R²: {r2:.4f}")
        logging.info(f"   Predicted range: {y_pred.min():.3f} - {y_pred.max():.3f}")
        
        return history, (X_test, y_test, y_pred)
    
    def predict_competitive_score(self, card_data):
        """Predict competitive score for a single card"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Extract features
        features = self.extract_features(card_data)
        features_df = pd.DataFrame([features])
        
        # Encode categorical features
        for feature in self.categorical_features:
            if feature in features_df.columns and feature in self.label_encoders:
                le = self.label_encoders[feature]
                features_df[feature] = le.transform(features_df[feature].astype(str))
        
        # Scale numerical features
        numerical_data = features_df[self.numerical_features].fillna(0)
        scaled_numerical = self.scaler.transform(numerical_data)
        features_df[self.numerical_features] = scaled_numerical
        
        # Predict
        score = self.model.predict(features_df[self.feature_columns])[0][0]
        return float(score)
    
    def score_all_cards(self, df):
        """Score all cards in the dataset"""
        logging.info("🎯 Scoring all cards for competitive viability...")
        
        # First pass: get all scores for even distribution
        all_scores = []
        for idx, row in df.iterrows():
            try:
                score = self.predict_competitive_score(row)
                all_scores.append(score)
            except Exception as e:
                logging.warning(f"Error scoring card {row.get('name', 'unknown')}: {e}")
                all_scores.append(0.0)
        
        # Second pass: categorize with even distribution
        scores = []
        for idx, row in df.iterrows():
            try:
                score = all_scores[idx]
                scores.append({
                    'id': row.get('id'),
                    'name': row.get('name'),
                    'supertype': row.get('supertype'),
                    'competitive_score': score,
                    'score_category': self.categorize_score(score, all_scores)
                })
            except Exception as e:
                logging.warning(f"Error processing card {row.get('name', 'unknown')}: {e}")
                scores.append({
                    'id': row.get('id'),
                    'name': row.get('name'),
                    'supertype': row.get('supertype'),
                    'competitive_score': 0.0,
                    'score_category': 'Unknown'
                })
        
        scores_df = pd.DataFrame(scores)
        
        # Show top competitive cards
        top_cards = scores_df.nlargest(20, 'competitive_score')
        logging.info("🏆 Top 20 Competitive Cards:")
        for _, card in top_cards.iterrows():
            logging.info(f"   {card['name']}: {card['competitive_score']:.3f} ({card['score_category']})")
        
        return scores_df
    
    def categorize_score(self, score, all_scores=None):
        """Categorize competitive score with even distribution if all_scores provided"""
        if all_scores is not None:
            # Sort all scores to find percentile cutoffs for even distribution
            sorted_scores = sorted(all_scores)
            n = len(sorted_scores)
            
            # Calculate percentile cutoffs for even distribution (20% each)
            s_cutoff = sorted_scores[int(0.8 * n)]  # Top 20%
            a_cutoff = sorted_scores[int(0.6 * n)]  # Next 20%
            b_cutoff = sorted_scores[int(0.4 * n)]  # Next 20%
            c_cutoff = sorted_scores[int(0.2 * n)]  # Next 20%
            # Bottom 20% is D-Tier
            
            if score >= s_cutoff:
                return "S-Tier"
            elif score >= a_cutoff:
                return "A-Tier"
            elif score >= b_cutoff:
                return "B-Tier"
            elif score >= c_cutoff:
                return "C-Tier"
            else:
                return "D-Tier"
        else:
            # Fallback to fixed thresholds
            if score >= 0.8:
                return "S-Tier"
            elif score >= 0.6:
                return "A-Tier"
            elif score >= 0.4:
                return "B-Tier"
            elif score >= 0.2:
                return "C-Tier"
            else:
                return "D-Tier"

def main():
    """Main function to run the neural network competitive scoring"""
    logging.info("🚀 Starting Neural Network Competitive Scoring")
    
    # Load data from database
    try:
        conn = sqlite3.connect("pokemon_cards.db")
        df = pd.read_sql_query("""
            SELECT * FROM card_info 
            WHERE supertype IN ('Pokémon', 'Trainer', 'Energy')
        """, conn)
        conn.close()
        logging.info(f"📊 Loaded {len(df)} cards from database")
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        return
    
    # Initialize scorer
    scorer = PokemonCardNeuralScorer()
    
    # Prepare data
    features_df = scorer.prepare_data(df)
    
    # Generate labels
    labels = scorer.generate_training_labels(df)
    
    # Train model
    history, (X_test, y_test, y_pred) = scorer.train_model(features_df, labels)
    
    # Score all cards
    scores_df = scorer.score_all_cards(df)
    
    # Save results to CSV
    output_file = f"competitive_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    scores_df.to_csv(output_file, index=False)
    logging.info(f"💾 Results saved to {output_file}")
    
    # Add competitive scores to SQL database
    try:
        conn = sqlite3.connect("pokemon_cards.db")
        cursor = conn.cursor()
        
        # Check if competitive_score column exists, add if not
        cursor.execute("PRAGMA table_info(cards)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'competitive_score' not in columns:
            cursor.execute("ALTER TABLE cards ADD COLUMN competitive_score REAL")
            logging.info("✅ Added competitive_score column to cards table")
        
        if 'competitive_tier' not in columns:
            cursor.execute("ALTER TABLE cards ADD COLUMN competitive_tier TEXT")
            logging.info("✅ Added competitive_tier column to cards table")
        
        # Update competitive scores in database
        updated_count = 0
        for _, row in scores_df.iterrows():
            cursor.execute("""
                UPDATE cards 
                SET competitive_score = ?, competitive_tier = ?
                WHERE id = ?
            """, (row['competitive_score'], row['score_category'], row['id']))
            updated_count += cursor.rowcount
        
        conn.commit()
        conn.close()
        logging.info(f"✅ Updated competitive scores for {updated_count} cards in database")
        
    except Exception as e:
        logging.error(f"❌ Error updating database: {e}")
    
    # Save model
    model_file = "competitive_scoring_model.h5"
    scorer.model.save(model_file)
    logging.info(f"💾 Model saved to {model_file}")
    
    # Show statistics
    logging.info("📊 Competitive Score Statistics:")
    logging.info(f"   Average Score: {scores_df['competitive_score'].mean():.3f}")
    logging.info(f"   Median Score: {scores_df['competitive_score'].median():.3f}")
    logging.info(f"   Score Range: {scores_df['competitive_score'].min():.3f} - {scores_df['competitive_score'].max():.3f}")
    logging.info(f"   Score Std: {scores_df['competitive_score'].std():.3f}")
    logging.info(f"   S-Tier Cards: {len(scores_df[scores_df['score_category'] == 'S-Tier'])}")
    logging.info(f"   A-Tier Cards: {len(scores_df[scores_df['score_category'] == 'A-Tier'])}")
    logging.info(f"   B-Tier Cards: {len(scores_df[scores_df['score_category'] == 'B-Tier'])}")
    logging.info(f"   C-Tier Cards: {len(scores_df[scores_df['score_category'] == 'C-Tier'])}")
    logging.info(f"   D-Tier Cards: {len(scores_df[scores_df['score_category'] == 'D-Tier'])}")
    
    logging.info("🎉 Neural Network Competitive Scoring Complete!")

if __name__ == "__main__":
    main()
