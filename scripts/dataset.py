from datasets import load_dataset
import wandb
import lightning as L
from omegaconf import DictConfig
import hydra
import pandas as pd


class DatasetUtilities:
    """
    A utility class for working with datasets.

    Args:
        name (str): The name of the dataset.

    Attributes:
        name (str): The name of the dataset.
        dataset: The loaded dataset.

    Methods:
        __init__(name: str): Initialize the DatasetUtilities class.
        process_dataset(): Process the dataset.
        export_to_csv(path: str): Export the dataset to a CSV file.
    """

    def __init__(self, name: str):
        """
        Initialize the DatasetUtilities class.

        Args:
            name (str): The name of the dataset.
        """
        self.name = name
        self.dataset = load_dataset(self.name)

    def process_dataset(self):
        """
        Process the dataset.

        This method processes the dataset by extracting the 'text' and 'prediction' data from the 'train' section of the dataset.
        It then creates a DataFrame with the extracted data and returns it.

        Returns:
            pandas.DataFrame: The processed dataset as a DataFrame with 'review' and 'rating' columns.
        """
        text = self.dataset["train"]["text"]
        label = self.dataset["train"]["prediction"]
        reviews = []
        ratings = []
        for i in range(len(text)):
            t = text[i]
            l = label[i][0]["label"]
            reviews.append(t)
            ratings.append(l)
        df = pd.DataFrame({"review": reviews, "rating": ratings})
        df = df.reset_index(drop=True)
        return df
    
    
    def export_to_csv(self, path: str):
        """
        Export the dataset to a CSV file.

        Args:
            path (str): The path to the CSV file.
        """
        df = self.process_dataset()
        return df.to_csv(path)


@hydra.main(version_base=None, config_path="../configs", config_name="dataset")
def main(cfg: DictConfig):
    dataset_utils = DatasetUtilities(cfg.topic_modelling_dataset.name)
    dataset_utils.export_to_csv(path=cfg.topic_modelling_dataset.processed_dataset)


main()
