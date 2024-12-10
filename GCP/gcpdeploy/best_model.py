import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """Load and return the dataset from the given filepath."""
    return pd.read_csv(filepath, index_col='date')

def split_time_series_data(df, target_column, test_size=0.1, val_size=0.1):
    """Split data into training, validation, and test sets."""
    df = df.sort_index()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(1-test_size), shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train, model_type='ridge'):
    """Train a model based on model type and return the trained model and its RMSE on the validation set."""
    if model_type == 'ridge':
        model = Ridge()
        param_grid = {'alpha': [0.05, 0.1, 0.2, 1.0]}
    elif model_type == 'svr':
        model = SVR()
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear'],
            'epsilon': [0.1, 0.2, 0.5]
        }
    grid_search = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=5) if model_type == 'ridge' else 3, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, -grid_search.best_score_**0.5

def select_best_model(models_metrics):
    """Select and return the best model based on RMSE."""
    best_model_name, best_rmse = None, float('inf')
    for model_name, rmse in models_metrics.items():
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = model_name
    return best_model_name, best_rmse

def main(args):
    """Main function to orchestrate the training and evaluation."""
    df = load_data(args.data_file)
    X_train, X_val, _, y_train, y_val, _ = split_time_series_data(df, 'close')

    models = {
        'ridge': train_model(X_train, y_train, 'ridge'),
        'svr': train_model(X_train, y_train, 'svr'),
    }
    models_metrics = {name: rmse for name, (_, rmse) in models.items()}
    best_model_name, best_rmse = select_best_model(models_metrics)

    logging.info(f"Best Model: {best_model_name} with RMSE: {best_rmse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate machine learning models.')
    parser.add_argument('--data_file', type=str, required=True, help='Filepath for the input data CSV file.')
    args = parser.parse_args()
    main(args)
