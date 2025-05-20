import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=42),
        'Lasso Regression': Lasso(random_state=42, max_iter=10000),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, eval_metric='rmse')
    }

    # GridSearchCV params for Ridge and Lasso
    ridge_params = {'alpha': [0.1, 1.0, 10.0]}
    lasso_params = {'alpha': [0.001, 0.01, 0.1]}

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'Ridge Regression':
            grid = GridSearchCV(model, ridge_params, cv=5, scoring='neg_mean_squared_error')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        elif name == 'Lasso Regression':
            grid = GridSearchCV(model, lasso_params, cv=5, scoring='neg_mean_squared_error')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_model = model

        preds = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2 Score': r2
        })

    return pd.DataFrame(results)
