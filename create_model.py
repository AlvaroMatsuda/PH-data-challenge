import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import pandas
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error


# ==========================
# Variables
# ==========================
# path to CSV with home sale data
SALES_PATH = "app/data/kc_house_data.csv" 

# path to CSV with demographics
DEMOGRAPHICS_PATH = "app/data/zipcode_demographics.csv"  

# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

# Directory where output artifacts will be saved
OUTPUT_DIR = "app/model"  

# Set random_state see
RANDOM_STATE_SEED = 42

# Basic model parameters
basic_model_params = {"n_jobs": -1}

# Model Parameter Search Space
fine_tune_params = {
    "knr__n_neighbors": [3, 5, 8, 12, 15, 20],
    "knr__weights": ["uniform", "distance"],
    "knr__leaf_size": [10, 20, 30, 40, 50]
}

# ===============================
# Helper Functions
# ===============================
def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv(demographics_path,
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y

def train_model(
    x_train: pandas.DataFrame,
    y_train: pandas.Series,
    model,
    model_params: dict
    ):
    """
    Create, train and evaluate model performance.

    Args:
        x_train: pandas.DataFrame containing training data.
        y_train: pandas.Series containing the target values of x_train.
        sklearn_model: A SKlearn model object
        model_params: Dictionary with model parameters.

    Returns:
        SKlearn pipeline object fitted with x_train and x_test data
    """
    # Creating the model pipeline
    model = (
        Pipeline([
            ("scaler", preprocessing.RobustScaler()),
            ("knr", model(**model_params))
            ])
        .fit(x_train, y_train)
        )

    return model

def evaluate_model(
    trained_model,
    x_test: pandas.DataFrame,
    y_test: pandas.Series,
    ):
    """
        Evaluate the model performance. Calculates MAPE, RMSE and R2.

        Args:
            trained_model: Fitted/trained SKlearn pipeline or model object.
            x_test: pandas.DataFrame containing test data.
            y_test: pandas.Series containing the target values of x_test.
        
        Return:
            Dictionary with model performance metrics
    """

    # Model Predictions
    y_pred = trained_model.predict(x_test)

    # Performance Metrics
    mape = float(mean_absolute_percentage_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    metrics = {"mape": mape, "rmse": rmse, "r2": r2, "mae": mae}

    return metrics


def random_search_fine_tune(
    model,
    params_search_space: dict[list],
    x_train: pandas.DataFrame,
    y_train: pandas.Series,
    ):
    """
    Fine-tune model performance. Apply SKLearn Random Search on the params_search_space.

    Args:
            model: SKlearn pipeline or model object.
            params_search_space: A Dictionary containing list of values to search from for each model parameter.
            x_train: pandas.DataFrame containing training data.
            y_train: pandas.Series containing the target values of x_train.
        
        Return:
            Best tuned model. It's a SKlearn pipeline or model object.
    """
    model_pipe = (
        Pipeline([
            ("scaler", preprocessing.RobustScaler()),
            ("knr", model)
            ])
        )

    random_search = RandomizedSearchCV(
        estimator=model_pipe,
        param_distributions=params_search_space,
        n_iter=10,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1,
        refit=True,
        cv=5,
        random_state=RANDOM_STATE_SEED
    )

    random_search.fit(x_train, y_train)

    tuned_model = model_pipe.set_params(**random_search.best_params_)

    tuned_model.fit(x_train, y_train)

    return tuned_model

def main():
    """Load data, train model, and export artifacts."""
    
    # Loading data
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    # Train-Test Split
    x_train, _x_test, y_train, _y_test = train_test_split(
        x, y, random_state=RANDOM_STATE_SEED)
    
    # Training Basic model
    model = train_model(x_train, y_train, neighbors.KNeighborsRegressor, basic_model_params)
    
    # Evaluate basic model
    metrics = evaluate_model(model, _x_test, _y_test)

    # Fine Tune model
    tuned_model = random_search_fine_tune(neighbors.KNeighborsRegressor(), fine_tune_params, x_train, y_train)

    # Evaluate tuned model
    tuned_metrics = evaluate_model(tuned_model, _x_test, _y_test)

    # Get the best model result
    if metrics["rmse"] < tuned_metrics["rmse"]:
        final_model = model
        final_metrics = metrics
        final_model_params = model[-1].get_params()
    else:
        final_model = tuned_model
        final_metrics = tuned_metrics
        final_model_params = tuned_model[-1].get_params()

    
    # enable autologging
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.sklearn.autolog()    

    with mlflow.start_run(run_name="best_model"):
        # Logging info to MLFLow
        mlflow.log_params(final_model_params)
        mlflow.log_metrics(final_metrics)
        mlflow.sklearn.log_model(final_model, "model")

    # Saving Artifacts
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(final_model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns), open(output_dir / "model_features.json", 'w'))
    json.dump(final_metrics, open(output_dir / "model_metrics.json", 'w'))

    # Printing for Sanity Check
    print("Model name:")
    print(final_model[-1])

    print("Model Metrics:")
    print(final_metrics)

if __name__ == "__main__":
    main()
