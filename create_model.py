import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import pandas
import numpy as np
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


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


def main():
    """Load data, train model, and export artifacts."""
    
    # Loading data
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    # Train-Test Split
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)

    # Train model

    # model = (
    #     pipeline.make_pipeline(
    #         preprocessing.RobustScaler(),
    #         neighbors.KNeighborsRegressor()
    #         )
    #     .fit(x_train, y_train)
    #     )
    
    model = (
        pipeline
        .make_pipeline(
            preprocessing.RobustScaler(),
            tree.DecisionTreeRegressor()
            )
        .fit(x_train, y_train)
        )

    # Model Performance
    y_pred = model.predict(_x_test)

    mape = float(mean_absolute_percentage_error(_y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(_y_test, y_pred)))
    r2 = float(r2_score(_y_test, y_pred))
    metrics = {"mape": mape, "rmse": rmse, "r2": r2}



    # Saving Artifacts
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns), open(output_dir / "model_features.json", 'w'))
    json.dump(metrics, open(output_dir / "model_metrics.json", 'w'))

    # Printing for Sanity Check
    print("Model name:")
    print(model[1])

    print("Model Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
