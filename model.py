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

def inference(csv_path: str = "winequality-red.csv"):
    """
       Receives a text promtp and audio path, and returns a synthesized audio file.
        Parameters:
        - text (str): Text prompt to be synthesized. Example: 'Hello, my dog is cooler than you!'.
        - audio_path (str) Optional: File path to save the synthesized audio. Example: 'bark_out.wav'.
        Returns:
        - bool: True if the audio file was saved successfully.
    """
    data_file = cached_download(
    hf_hub_url(REPO_ID, "winequality-red.csv")
    )
    winedf = pd.read_csv(data_file, sep=";")
    X = winedf.drop(["quality"], axis=1)
    model = load_model()
    labels = model.predict(X[:3])
    return str(labels)