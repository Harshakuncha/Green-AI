import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3


def load_and_prepare_data(combined_df):
    """
    Load and label the results from different feature selection techniques.
    """
    num_rows = len(combined_df)
    group_size = 30
    group_labels = ['Processed_data','PCA_data','Top3_data']

    # Repeat each label `group_size` times, then truncate to the number of rows
    combined_df['Technique'] = np.repeat(group_labels, group_size)[:num_rows]

    return combined_df


def plot_comparisons(combined_df, save_path):
    
    # Set the figure size for the plots
    plt.figure(figsize=(12, 8))

    # Plot RMSE comparison
    plt.subplot(1, 2, 1)
    sns.barplot(data=combined_df, x='Technique', y='RMSE', hue='Regressor')
    plt.title('RMSE Comparison by Feature Selection Technique')
    plt.ylabel('RMSE')
    plt.xlabel('Feature Selection Technique')
    plt.xticks(rotation=45)

    # Plot R² comparison
    plt.subplot(1, 2, 2)
    sns.barplot(data=combined_df, x='Technique', y='R² Score', hue='Regressor')
    plt.title('R² Score Comparison by Feature Selection Technique')
    plt.ylabel('R² Score')
    plt.xlabel('Feature Selection Technique')
    plt.xticks(rotation=45)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plots as a PNG file
    plt.savefig(save_path)

    # Display the plots
    plt.show()







