# Imports
from Preprocess_package import load_data, process_llm_models, select_top_features, apply_pca
from Modelling_Package import evaluate_regressors, plot_comparisons ,load_and_prepare_data
import pandas as pd

# Paths
data = "./Dataset/Preprocessed.csv"
output = '/Users/bhargav/Documents/Green-AI/Output/'

# Loading Preprocessed MELODI dataset
df = load_data(data)

# Creating a dictionary of individual LLM and respective energy consumption
Orginal_results = process_llm_models(df)

# Performing PCA based on the analysis from checkpoint-1
PCA_results = apply_pca(Orginal_results,n_components=10) 

# Selection of Custom top features from random forest feature importance 
Top_Features = select_top_features(Orginal_results,3)

# Prediction
Orginal_results_evaluation = evaluate_regressors(Orginal_results)
PCA_results_evaluation = evaluate_regressors(PCA_results)
Top_Features_evaluation = evaluate_regressors(Top_Features)
combined_df = pd.concat([Orginal_results_evaluation, PCA_results_evaluation, Top_Features_evaluation], ignore_index=True)
combined_df.to_csv(output+"Metrics.csv",index=False)

# Plots for comparision
Modified_df = load_and_prepare_data(combined_df)
plot_comparisons(combined_df, output+"comparision_plots.png")
