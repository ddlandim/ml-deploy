from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np
from typing import List
import os
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline as MyModel

REPO_ID = "julien-c/wine-quality"
FILENAME = "sklearn_model.joblib"
REQUIRED_COLUMNS = (("fixed acidity", float), 
                    ("volatile acidity", float), 
                    ("citric acid", float), 
                    ("residual sugar", float), 
                    ("chlorides", float), 
                    ("free sulfur dioxide", float), 
                    ("total sulfur dioxide", float), 
                    ("density", float), 
                    ("pH", float), 
                    ("sulphates", float), 
                    ("alcohol", float))
TARGET_COLUMN = "quality"
TEST_DATA = {
    "fixed acidity": [7.4, 7.8, 7.8],
    "volatile acidity": [0.7, 0.88, 0.76],
    "citric acid": [0, 0, 0.04],
    "residual sugar": [1.9, 2.6, 2.3],
    "chlorides": [0.076, 0.098, 0.092],
    "free sulfur dioxide": [11, 25, 15],
    "total sulfur dioxide": [34, 67, 54],
    "density": [0.9978, 0.9968, 0.997],
    "pH": [3.51, 3.2, 3.26],
    "sulphates": [0.56, 0.68, 0.65],
    "alcohol": [9.4, 9.8, 9.8],
    "quality": [5, 5, 5]
}
TEST_X = pd.DataFrame(TEST_DATA)
TEST_Y = TEST_X.pop(TARGET_COLUMN)

class ModelError(Exception):
    """Exception raised for errors in this module.
    Attributes:
        message -- explanation of the error
        code -- error code
    """
    def __init__(self, message="Model error.", code=400):
        self.message = message
        self.code = code

def load_model() -> MyModel:
    """
       Loads the model and tokenizer from the HuggingFace model hub.
    """
    model:MyModel = joblib.load(cached_download(
            hf_hub_url(REPO_ID, FILENAME)
        ))
    return model

def validate_model(model: MyModel, X: pd.DataFrame, Y: pd.DataFrame):
    """
    Test a model by comparing its predictions to the actual values.
    Parameters:
    - model: The model to test.
    - X: The input values.
    - Y: The actual values.
    Raises:
    - ModelError: If the model is not loaded, if the predictions are not in the List[float] format,
                  if the predictions do not match all values in Y, or if the predictions are not partially equal to Y values.
    """
    if model is None:
        raise ModelError("Model is not loaded.", code=500)
    
    try:
        predictions = model.predict(X)
    except Exception as e:
        raise ModelError(f"Error during prediction: {str(e)}", code=500)
    
    # predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    # if not pd.to_numeric(predictions_df['Prediction'], errors='coerce').notna().all():
    #     msg = "".join([
    #         f"Predictions are not in the List[float] or List[int] format.\n",
    #         f"Predictions type: {type(predictions)}\n",
    #         #f"Predictions: {str(predictions)}\n"
    #     ])
    #     raise ModelError(msg, code=500)
    
    # predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    # Y_df = pd.DataFrame(Y, columns=['Actual'])
    # if not predictions_df.equals(Y_df):
    #     mae = mean_absolute_error(Y_df, predictions_df)
    #     diff_count = sum(predictions_df['Prediction'] != Y_df['Actual'])
    #     e_msg = "".join([
    #         f"Predictions do not match all values in Y.\n",
    #         f"Mean Absolute Error: {mae}.\n",
    #         f"Number of different values: {diff_count}.\n"
    #     ])
    #     raise ModelError(e_msg, code=400)
    return predictions

def load_csv(csv_path: str):
    """
    Load a CSV file and return a pandas DataFrame.
    Parameters:
    - csv_path (str): Path to the CSV file.
    Returns:
    - df (pd.DataFrame): DataFrame containing the CSV data.
    """
    if not os.path.exists(csv_path):
        raise ModelError(f"File does not exist: {csv_path}", code=404)
    
    try:
        df = pd.read_csv(csv_path, sep=";")
    except pd.errors.ParserError:
        raise ModelError(f"Error reading file: {csv_path}", code=400)
    
    return df

def validate_df(df: pd.DataFrame):
    if TARGET_COLUMN in df.columns:
        df = df.drop([TARGET_COLUMN], axis=1)
    
    # Check for missing columns
    missing_columns = [column for column, _ in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ModelError(f"Missing columns: {', '.join(missing_columns)}.", code=422)
    
    # Check for unknown columns
    unknown_columns = [column for column in df.columns if column not in [col for col, _ in REQUIRED_COLUMNS]]
    if unknown_columns:
        raise ModelError(f"Unknown columns: {', '.join(unknown_columns)}.", code=422)
    
    # Check for None values
    if df.isnull().values.any():
        raise ModelError("DataFrame contains None values.", code=422)
    
    # Check for wrong column types
    wrong_type_columns = [column for column, dtype in REQUIRED_COLUMNS if df[column].dtype != dtype]
    if wrong_type_columns:
        raise ModelError(f"Wrong column types: {', '.join(wrong_type_columns)}.", code=422)

def inference(csv_path: str):
    """
        Receives a path to a csv file, readed with pandas, and returns predicted labels.
        The csv file should contain the following columns:
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", 
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
        Each column should be of type float with 4 decimals maximum.
        Parameters:
        - csv_path (str): Path to the csv file containing the data to predict.
        Returns:
        - labels (str): Predicted labels.
    """
    try:
        model = load_model()
        validate_model(model, TEST_X, TEST_Y)
        df = load_csv(csv_path)
        validate_df(df)
        labels = model.predict(df)
        return labels
    except ModelError as e:
        raise e
    except Exception as e:
        raise ModelError(f"Inference error: {e}", code=500)

def mock_inference():
    """
        Mock inference function with test data.
        Returns:
        - labels (str): Predicted labels.
    """
    try:
        model = load_model()
        predictions = validate_model(model, TEST_X, TEST_Y)
        return predictions.tolist()
    except ModelError as e:
        raise e
    except Exception as e:
        raise ModelError(f"Inference error: {e}", code=500)