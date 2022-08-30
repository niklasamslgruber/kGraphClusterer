from typing import Optional
import pandas as pd
from os.path import exists
from os import mkdir
from enum import Enum, unique


class DataProcessor:
    @unique
    class Dataset(Enum):
        ADULTS = "data/adults/adults.csv"
        BANK_CLIENTS = "data/banks/banks.csv"

        def getDirectory(self):
            return "/".join(self.value.split("/")[:-1])

        def createDirectoryIfNotExists(self):
            if not exists(self.getDirectory()):
                mkdir(self.getDirectory())

        def getEdgePath(self):
            return f"{self.getDirectory()}/edges.csv"

    @staticmethod
    def loadData(dataset: Dataset) -> Optional[pd.DataFrame]:
        if not exists(dataset.value):
            print(f"Data for dataset {dataset.name} does not exist. Starting to process it now...")
            match dataset:
                case DataProcessor.Dataset.BANK_CLIENTS:
                    DataProcessor.__processBankData()
                case DataProcessor.Dataset.ADULTS:
                    DataProcessor.__processAdultData()
                case _:
                    return None

        frame = pd.read_csv(dataset.value)
        DataProcessor.checkEdgeFile(dataset)
        return frame

    @staticmethod
    def checkEdgeFile(dataset: Dataset):
        if not exists(dataset.getEdgePath()):
            print(f"No edge file (edges.csv) found for {dataset.name}")

    # Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    @staticmethod
    def __processBankData() -> pd.DataFrame:
        dataset = DataProcessor.Dataset.BANK_CLIENTS
        frame = pd.read_csv("data/raw/bank/bank-full.csv", sep=";")

        for column in ["contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]:
            frame = frame.drop(column, axis=1)

        dataset.createDirectoryIfNotExists()

        frame.to_csv(dataset.value, index_label="id")
        return frame

    # Source: https://archive.ics.uci.edu/ml/datasets/Adult
    @staticmethod
    def __processAdultData() -> pd.DataFrame:
        dataset = DataProcessor.Dataset.ADULTS
        frame = pd.read_csv("data/raw/adult/adult.data", header=None)
        frame.columns = ["age", "workclass", "finalWeight", "education", "education-num", "martial-status",
                         "occupation", "relationship", "race", "sex", "captial-gain", "capital-loss", "hours-per-week",
                         "native-country", "income"]

        dataset.createDirectoryIfNotExists()

        frame.to_csv(dataset.value, index_label="id")
        return frame
