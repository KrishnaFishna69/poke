import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re
import logging
import sqlite3

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('price_prediction.log'),
        logging.StreamHandler()
    ]
)

def load_data_from_sql():
    """
    Load data from SQLite database instead of CSV
    """
    try:
        conn = sqlite3.connect("../pokemon_cards.db")
        
        # Get the most recent price column from price_history table
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(price_history)")
        columns = [col[1] for col in cursor.fetchall()]
        logging.info(f"Found {len(columns)} columns in price_history table")
        
        price_columns = [col for col in columns if col.startswith('cardMarketPrice-')]
        logging.info(f"Found {len(price_columns)} price columns: {price_columns[:5]}...")
        
        if not price_columns:
            logging.error("No price columns found in price_history table")
            return None, None
        
        # Use the most recent price column (filter out test/simulation columns)
        valid_price_columns = [col for col in price_columns if not any(suffix in col for suffix in ['-test', '-simulation'])]
        latest_price_col = sorted(valid_price_columns)[-1]
        logging.info(f"Using price column: {latest_price_col}")
        
        # Load data with all features including art scores and competitive scores
        query = f"""
            SELECT c.*, 
                   COALESCE(c.art_score_0_10, 0) as art_score,
                   COALESCE(0, 0) as competitive_score,
                   COALESCE('Unknown', 'Unknown') as competitive_tier,
                   COALESCE(ph."{latest_price_col}", 0) as current_price
            FROM cards c
            LEFT JOIN price_history ph ON c.id = ph.id
            WHERE c.supertype IN ('Pokémon', 'Trainer', 'Energy')
            AND ph."{latest_price_col}" IS NOT NULL 
            AND ph."{latest_price_col}" > 0
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logging.info(f"Loaded {len(df)} cards with price data from database")
        return df, latest_price_col
        
    except Exception as e:
        logging.error(f"Error loading data from database: {e}")
        return None, None

def extract_set_id(set_info):
    """
    Extract the set ID from the set information string.
    """
    try:
        if pd.isna(set_info):
            return None
        
        # Look for pattern like "Set(id='xy5', ...)" and extract 'xy5'
        match = re.search(r"Set\(id='([^']*)'", str(set_info))
        if match:
            return match.group(1)
        
        # Fallback: try to extract any alphanumeric code
        match = re.search(r"id='([a-zA-Z0-9]+)'", str(set_info))
        if match:
            return match.group(1)
        
        return None
    except Exception as e:
        logging.warning(f"Error extracting set ID from {set_info}: {str(e)}")
        return None

def extract_standard_legality(legalities_str):
    """
    Extract Standard legality from the legalities string.
    """
    try:
        if pd.isna(legalities_str):
            return 'Unknown'
        
        # Look for standard legality in the string
        match = re.search(r"standard='([^']*)'", str(legalities_str))
        if match:
            legality = match.group(1)
            if legality == 'None':
                return 'Not Legal'
            return legality
        return 'Unknown'
    except Exception as e:
        logging.warning(f"Error extracting standard legality from {legalities_str}: {str(e)}")
        return 'Unknown'

def preprocess_features(df):
    """
    Preprocess the features for the XGBoost model, now including art scores and competitive scores.
    """
    logging.info("Starting feature preprocessing...")
    
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # 1. Extract set ID from set column
    logging.info("Extracting set IDs...")
    df_processed['set_id'] = df_processed['set'].apply(extract_set_id)
    
    # 2. Extract Standard legality
    logging.info("Extracting Standard legality...")
    df_processed['standard_legal'] = df_processed['legalities'].apply(extract_standard_legality)
    
    # 3. Handle subtypes (convert list to string)
    logging.info("Processing subtypes...")
    df_processed['subtypes_str'] = df_processed['subtypes'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) and len(x) > 0 else 'None'
    )
    
    # 4. Handle types (convert list to string)
    logging.info("Processing types...")
    df_processed['types_str'] = df_processed['types'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) and len(x) > 0 else 'None'
    )
    
    # 5. Process competitive tier (convert to numeric)
    logging.info("Processing competitive tiers...")
    tier_mapping = {
        'S-Tier': 5, 'A-Tier': 4, 'B-Tier': 3, 'C-Tier': 2, 'D-Tier': 1, 'Unknown': 0
    }
    df_processed['competitive_tier_numeric'] = df_processed['competitive_tier'].map(tier_mapping)
    
    # 6. Select features for the model (now including art and competitive scores)
    feature_columns = [
        'supertype', 'subtypes_str', 'types_str', 'set_id', 
        'artist', 'rarity', 'standard_legal', 'popularity_votes', 'avg_votes',
        'art_score', 'competitive_score', 'competitive_tier_numeric'  # New features
    ]
    
    # 7. Handle missing values
    for col in feature_columns:
        if col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].fillna('Unknown')
            else:
                df_processed[col] = df_processed[col].fillna(0)
    
    logging.info(f"Feature preprocessing complete. Using {len(feature_columns)} features.")
    return df_processed, feature_columns

def encode_features(df, feature_columns):
    """
    Encode categorical features using LabelEncoder.
    """
    logging.info("Encoding categorical features...")
    
    encoders = {}
    df_encoded = df.copy()
    
    for col in feature_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            logging.info(f"Encoded {col} with {len(le.classes_)} unique values")
    
    return df_encoded, encoders

def prepare_data(df, target_column='cardMarketPrice-06-19-25'):
    """
    Prepare the data for training the XGBoost model.
    """
    logging.info("Preparing data for XGBoost model...")
    
    # Preprocess features
    df_processed, feature_columns = preprocess_features(df)
    
    # Encode features
    df_encoded, encoders = encode_features(df_processed, feature_columns)
    
    # Create feature matrix
    encoded_columns = [f'{col}_encoded' for col in feature_columns]
    X = df_encoded[encoded_columns]
    
    # Handle target variable
    if target_column in df_encoded.columns:
        # Remove rows with missing target values
        valid_mask = df_encoded[target_column].notna()
        X = X[valid_mask]
        y = df_encoded[target_column][valid_mask]
        
        logging.info(f"Final dataset: {len(X)} samples with {len(X.columns)} features")
        logging.info(f"Target variable range: {y.min():.2f} to {y.max():.2f}")
        
        return X, y, encoders, feature_columns
    else:
        logging.error(f"Target column '{target_column}' not found in dataset")
        return None, None, None, None

def train_xgboost_model(X, y, test_size=0.2, random_state=42):
    """
    Train an XGBoost model for price prediction.
    """
    logging.info("Training XGBoost model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Log results
    logging.info("Model Performance:")
    logging.info(f"Train MSE: {train_mse:.2f}")
    logging.info(f"Test MSE: {test_mse:.2f}")
    logging.info(f"Train MAE: {train_mae:.2f}")
    logging.info(f"Test MAE: {test_mae:.2f}")
    logging.info(f"Train R²: {train_r2:.4f}")
    logging.info(f"Test R²: {test_r2:.4f}")
    
    return model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from the XGBoost model.
    """
    logging.info("Plotting feature importance...")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for easier plotting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def plot_predictions(y_true, y_pred, title):
    """
    Plot actual vs predicted values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def export_predictions_csv(df, y_true, y_pred, feature_columns, encoders, filename='price_predictions.csv'):
    """
    Export a CSV with card predictions, actual prices, and residuals.
    """
    logging.info("Exporting predictions CSV...")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'name': df['name'],
        'id': df['id'],
        'actual_price': y_true,
        'predicted_price': y_pred,
        'residual': y_true - y_pred,
        'abs_residual': abs(y_true - y_pred)
    })
    
    # Add feature columns for reference
    for col in feature_columns:
        if col in df.columns:
            results_df[f'feature_{col}'] = df[col]
    
    # Sort by absolute residual (largest errors first)
    results_df = results_df.sort_values('abs_residual', ascending=False)
    
    # Save to CSV
    results_df.to_csv(filename, index=False)
    logging.info(f"Predictions exported to {filename}")
    
    return results_df

def show_top_residuals(results_df, n=10):
    """
    Display the top N cards with largest residuals.
    """
    logging.info(f"Top {n} cards with largest residuals:")
    
    top_residuals = results_df.head(n)
    
    print(f"\nTop {n} Cards with Largest Prediction Errors:")
    print("=" * 80)
    for idx, row in top_residuals.iterrows():
        print(f"{row['name']} ({row['id']})")
        print(f"  Actual: ${row['actual_price']:.2f}")
        print(f"  Predicted: ${row['predicted_price']:.2f}")
        print(f"  Residual: ${row['residual']:.2f}")
        print(f"  Abs Residual: ${row['abs_residual']:.2f}")
        print("-" * 40)
    
    return top_residuals

def visualize_decision_tree(model, feature_names, max_depth=3):
    """
    Visualize the decision tree structure using XGBoost's built-in plotting.
    """
    logging.info("Visualizing decision tree structure...")
    
    try:
        # Create a smaller tree for visualization (first few trees)
        plt.figure(figsize=(20, 12))
        
        # Plot the first tree in the ensemble
        xgb.plot_tree(model, num_trees=0, rankdir='TB', 
                     feature_names=feature_names, 
                     max_depth=max_depth)
        
        plt.title(f"XGBoost Decision Tree (Tree 0, Max Depth: {max_depth})")
        plt.tight_layout()
        plt.savefig('decision_tree_structure.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also create a feature importance plot with more detail
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=20, 
                           feature_names=feature_names,
                           importance_type='gain')
        plt.title("XGBoost Feature Importance (Gain)")
        plt.tight_layout()
        plt.savefig('feature_importance_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logging.info("Decision tree visualizations saved to decision_tree_structure.png and feature_importance_detailed.png")
        
    except Exception as e:
        logging.warning(f"Could not create decision tree visualization: {str(e)}")
        logging.info("This might be due to XGBoost version or system limitations")

def analyze_feature_contributions(model, X_test, feature_names, n_samples=5):
    """
    Analyze how individual features contribute to predictions.
    """
    logging.info("Analyzing feature contributions...")
    
    try:
        # Get feature contributions for a few sample predictions
        # Use the newer XGBoost API
        contributions = model.predict(X_test[:n_samples], pred_contribs=True)
        
        print(f"\nFeature Contributions for {n_samples} Sample Predictions:")
        print("=" * 80)
        
        for i in range(n_samples):
            print(f"\nSample {i+1}:")
            print("-" * 40)
            
            # Get the feature contributions for this sample
            sample_contribs = contributions[i]
            
            # Create a list of (feature_name, contribution) pairs
            feature_contribs = list(zip(feature_names, sample_contribs[:-1]))  # Exclude bias term
            
            # Sort by absolute contribution
            feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Print top contributing features
            for feature, contrib in feature_contribs[:10]:
                print(f"  {feature}: {contrib:+.4f}")
            
            print(f"  Bias: {sample_contribs[-1]:+.4f}")
            print(f"  Total Prediction: {sum(sample_contribs):.4f}")
            
    except Exception as e:
        logging.warning(f"Could not analyze feature contributions: {str(e)}")
        logging.info("This might be due to XGBoost version differences")
        
        # Fallback: show feature importance instead
        print(f"\nFeature Importance Analysis (Fallback):")
        print("=" * 80)
        
        # Get feature importance scores
        importance_scores = model.feature_importances_
        feature_importance = list(zip(feature_names, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 10 Most Important Features:")
        for feature, importance in feature_importance[:10]:
            print(f"  {feature}: {importance:.4f}")

def main():
    """
    Main function to run the price prediction pipeline.
    """
    logging.info("Starting price prediction pipeline...")
    
    # Load the dataset
    df, latest_price_col = load_data_from_sql()
    
    if df is None:
        logging.error("Failed to load data from database. Exiting.")
        return
    
    # Prepare data
    X, y, encoders, feature_columns = prepare_data(df, target_column='current_price')
    
    if X is None or y is None:
        logging.error("Failed to prepare data. Exiting.")
        return
    
    # Train model
    model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_xgboost_model(X, y)
    
    # Plot results
    encoded_columns = [f'{col}_encoded' for col in feature_columns]
    feature_importance_df = plot_feature_importance(model, encoded_columns)
    
    plot_predictions(y_train, y_pred_train, "Training Set: Actual vs Predicted")
    plot_predictions(y_test, y_pred_test, "Test Set: Actual vs Predicted")
    
    # Export predictions CSV
    # We need to get the original dataframe rows that correspond to our test set
    # Get the indices of the test set
    test_indices = X_test.index
    df_test = df.iloc[test_indices]
    
    results_df = export_predictions_csv(df_test, y_test, y_pred_test, feature_columns, encoders)
    
    # Show top residuals
    top_residuals = show_top_residuals(results_df, n=10)
    
    # Visualize decision tree
    visualize_decision_tree(model, encoded_columns, max_depth=4)
    
    # Analyze feature contributions
    analyze_feature_contributions(model, X_test, encoded_columns, n_samples=5)
    
    # Save model and encoders
    import joblib
    joblib.dump(model, 'price_prediction_model.pkl')
    joblib.dump(encoders, 'price_prediction_encoders.pkl')
    
    logging.info("Model and encoders saved to price_prediction_model.pkl and price_prediction_encoders.pkl")
    logging.info("Feature importance plot saved to feature_importance.png")
    logging.info("Decision tree visualizations saved")
    
    # Print feature importance summary
    print("\nFeature Importance Summary:")
    print(feature_importance_df.to_string(index=False))
    
    # Print summary statistics
    print(f"\nPrediction Summary Statistics:")
    print(f"Mean Absolute Error: ${results_df['abs_residual'].mean():.2f}")
    print(f"Median Absolute Error: ${results_df['abs_residual'].median():.2f}")
    print(f"Standard Deviation of Residuals: ${results_df['residual'].std():.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred_test):.4f}")
    
    logging.info("Price prediction pipeline completed successfully!")

if __name__ == "__main__":
    main() 