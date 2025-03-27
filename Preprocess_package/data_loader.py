import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file and returns it as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Function to process raw data for each LLM model
def process_llm_models(df):
    """
    Process LLM models and extract raw data (without PCA).
    
    Parameters:
    - df: pandas DataFrame containing the data.
    
    Returns:
    - results: dict, containing raw data for each model.
    """
    
    # Get unique LLM model names
    llm_models = df['model_name'].unique()

    # Initialize dictionaries to store results
    results = {}

    # Loop through each model
    for model in llm_models:
        print(f"Processing {model}...")

        # Filter rows for this LLM model
        df_llm = df[df["model_name"] == model].copy()

        # Split features and target
        X = df_llm.drop(columns=["energy_consumption_llm", "model_name"])  # exclude LLM name & target
        y = df_llm['energy_consumption_llm']

        # Store original (non-PCA) version
        raw_data = {"X": X, "y": y}

        # Store raw data in the results
        results[model] = {"raw_data": raw_data}

    print("✅ Finished processing datasets for all LLM models.")
    return results


# Function to apply PCA to the raw data for each model
def apply_pca(results, n_components=10):
    """
    Apply PCA to raw data and return consistent structure with keys 'X' and 'y'.
    """
    pca_results = {}

    for model, data in results.items():
        print(f"🧪 Applying PCA to {model}...")

        # Extract raw data
        X = data["raw_data"]["X"]
        y = data["raw_data"]["y"]

        # Perform PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Store PCA result in the expected format
        X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        pca_results[model] = {
            "raw_data": {
                "X": X_pca_df,
                "y": y
            }
        }

    print("✅ Finished applying PCA to all models.")
    return pca_results


def select_top_features(results, num_features):
    """
    Select top 'num_features' features using Random Forest for each model,
    and return actual values (X and y) for those features.

    Parameters:
    - results: dict, raw data for each model.
    - num_features: int, number of top features to select.

    Returns:
    - top_features: dict, with model -> {"raw_data": {"X": top features DataFrame, "y": target Series}}.
    """
    top_features = {}

    for model, data in results.items():
        print(f"🔍 Selecting top {num_features} features for {model}...")

        # Extract raw data
        X = data["raw_data"]["X"]
        y = data["raw_data"]["y"]

        # Train Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get sorted feature importances
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values(by='importance', ascending=False)

        # Get names of top N features
        top_feature_names = feature_importance_df.head(num_features)['feature'].tolist()

        # Get corresponding subset of X
        X_top = X[top_feature_names]

        # Store the subset and y under 'raw_data'
        top_features[model] = {
            "raw_data": {
                "X": X_top,
                "y": y
            }
        }

    print(f"\n✅ Finished selecting top {num_features} features for all models.")
    return top_features
