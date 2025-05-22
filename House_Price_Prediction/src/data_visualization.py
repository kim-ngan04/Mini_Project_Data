import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('results/charts/correlation_heatmap.png')
    plt.close()

def plot_model_comparison(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='MAE', data=results_df, color='blue', alpha=0.6, label='MAE')
    sns.barplot(x='Model', y='RMSE', data=results_df, color='red', alpha=0.4, label='RMSE')
    plt.legend()
    plt.title('Model Comparison: MAE and RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/charts/model_comparison.png')
    plt.close()
