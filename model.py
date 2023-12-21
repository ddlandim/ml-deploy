from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

REPO_ID = "julien-c/wine-quality"
FILENAME = "sklearn_model.joblib"

def load_model():
    """
       Loads the model and tokenizer from the HuggingFace model hub.
    """
    model = joblib.load(cached_download(
            hf_hub_url(REPO_ID, FILENAME)
        ))
    return model

def inference(csv_path: str = None):
    """
       Receives a text promtp and audio path, and returns a synthesized audio file.
        Parameters:
        - csv_path (str): Path to the csv file containing the data to predict.
        Returns:
        - labels (str): Predicted labels.
    """
    if not csv_path:
        data_file = cached_download(
        hf_hub_url(REPO_ID, "winequality-red.csv")
        )
        winedf = pd.read_csv(data_file, sep=";")
    else:
        winedf = pd.read_csv(csv_path, sep=";")
    X = winedf.drop(["quality"], axis=1)
    model = load_model()
    labels = model.predict(X[:3])
    return str(labels)