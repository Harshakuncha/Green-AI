# standard
import os
import pandas as pd

# Machine Learning Model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Training
from sklearn.model_selection import train_test_split

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_regressors(raw_results):
    """
    Evaluate multiple regressors on raw data and save results as a CSV in the Output folder.

    Parameters:
    - raw_results: dict, contains model_name -> {"raw_data": {"X": features, "y": target}}.
    - output_path: str, path to save the resulting CSV.

    Returns:
    - results_df: pandas DataFrame with evaluation metrics.
    """
    # Define regressors
    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(),
        "SVM": SVR(),
        "KNN": KNeighborsRegressor()
    }

    # Store results
    results = []

    # Evaluate each model
    for model_name, data in raw_results.items():
        print(f"\n📊 Processing dataset: {model_name}")
        
        X = data["raw_data"]["X"]
        y = data["raw_data"]["y"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for reg_name, reg in regressors.items():
            print(f"⚙️  Training {reg_name} on {model_name}...")
            reg.fit(X_train, y_train)
            print(f"🔍 Evaluating {reg_name} on {model_name}...")
            y_pred = reg.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({
                "Dataset": model_name,
                "Regressor": reg_name,
                "RMSE": round(rmse, 4),
                "R² Score": round(r2, 4),
            })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    return results_df
