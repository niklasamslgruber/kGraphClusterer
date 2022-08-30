from typing import Optional
import pandas as pd
from os.path import exists
from dataHandler.datasets import Datasets


class DataProcessor:

    @staticmethod
    def loadFeatures(dataset: Datasets) -> Optional[pd.DataFrame]:
        if not exists(dataset.value):
            print(f"Data for dataset {dataset.name} does not exist. Starting to process it now...")
            match dataset:
                case Datasets.BANK_CLIENTS:
                    DataProcessor.__processBankData()
                case Datasets.ADULTS:
                    DataProcessor.__processAdultData()
                case _:
                    return None

        frame = pd.read_csv(dataset.value, index_col="id")
        DataProcessor.__checkEdgeFile(dataset)
        return frame

    @staticmethod
    def loadEdges(dataset: Datasets):
        if DataProcessor.__checkEdgeFile(dataset):
            frame = pd.read_csv(dataset.getEdgePath(), header=None)
            frame.columns = ["node1", "node2"]
            return frame

    # HELPER

    @staticmethod
    def __checkEdgeFile(dataset: Datasets):
        if not exists(dataset.getEdgePath()):
            print(f"No edge file (edges.csv) found for {dataset.name}")
            return False
        return True

    # Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    @staticmethod
    def __processBankData() -> pd.DataFrame:
        dataset = Datasets.BANK_CLIENTS
        frame = pd.read_csv("../data/raw/bank/bank-full.csv", sep=";")

        for column in ["contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]:
            frame = frame.drop(column, axis=1)

        dataset.createDirectoryIfNotExists()

        frame.to_csv(dataset.value, index_label="id")
        return frame

    # Source: https://archive.ics.uci.edu/ml/datasets/Adult
    @staticmethod
    def __processAdultData() -> pd.DataFrame:
        dataset = Datasets.ADULTS
        frame = pd.read_csv("../data/raw/adult/adult.data", header=None)
        frame.columns = ["age", "workclass", "finalWeight", "education", "education-num", "martial-status",
                         "occupation", "relationship", "race", "sex", "captial-gain", "capital-loss", "hours-per-week",
                         "native-country", "income"]

        dataset.createDirectoryIfNotExists()

        frame.to_csv(dataset.value, index_label="id")
        return frame
