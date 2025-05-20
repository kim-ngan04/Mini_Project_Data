from src.preprocessing import load_and_preprocess_data
from src.modeling import train_and_evaluate_models
from src.visualization import plot_correlation_heatmap, plot_model_comparison

import pandas as pd

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, full_data = load_and_preprocess_data('data/data.csv')

    # EDA visualization
    plot_correlation_heatmap(full_data)
    
    # Train models and get results
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Save results table
    results.to_csv('results/model_metrics.csv', index=False)

    # Plot model comparison chart
    plot_model_comparison(results)

    print("Training and evaluation done! Check 'results/' folder for outputs.")

if __name__ == '__main__':
    main()
