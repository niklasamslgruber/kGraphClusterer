from typing import Optional
import pandas as pd
from os.path import exists
from constants import RANDOM_SEED
from dataHandler.datasets import Datasets


class DataProcessor:

    @staticmethod
    def loadFeatures(dataset: Datasets, threshold: int) -> Optional[pd.DataFrame]:
        if not exists(dataset.getFeaturePath()):
            print(f"Data for dataset {dataset.name} does not exist. Starting to process it now...")
            match dataset:
                case Datasets.BANK_CLIENTS:
                    DataProcessor.__processBankData()
                case Datasets.ADULTS:
                    DataProcessor.__processAdultData()
                case _:
                    return None

        frame = pd.read_csv(dataset.getFeaturePath(), index_col="id")
        if threshold < len(frame):
            features = frame.sample(threshold, random_state=RANDOM_SEED)
        else:
            features = frame
        features = DataProcessor.loadAssociations(dataset, threshold, features)
        return features

    @staticmethod
    def loadAssociations(dataset: Datasets, threshold: int, features: pd.DataFrame):
        matched_features: pd.DataFrame
        if not exists(dataset.getAssociationPath(threshold)):
            matched_features = DataProcessor.__generateAssociations(dataset, threshold, features)
        else:
            matched_features = pd.read_csv(dataset.getAssociationPath(threshold))

        features["transactionID"] = matched_features["transactionID"].values

        # Verify that IDs and transactionIDs are correctly mapped
        for (index, feature) in features.iterrows():
            assert [feature["transactionID"]] == list(matched_features[matched_features["id"] == index]["transactionID"])

        return features

    @staticmethod
    def __generateAssociations(dataset: Datasets, threshold: int, features: pd.DataFrame):
        edges = DataProcessor.loadEdges(dataset, threshold)
        node_ids = list(edges["node1"].unique())

        assert len(node_ids) == threshold

        index = features.index

        associations = pd.DataFrame()
        associations["id"] = index
        associations["transactionID"] = node_ids

        associations.to_csv(dataset.getAssociationPath(threshold), index=False)

        for (index, row) in associations.iterrows():
            assert len(associations[associations["id"] == row["id"]]) == 1
            assert len(associations[associations["transactionID"] == row["transactionID"]]) == 1

        return associations

    @staticmethod
    def loadEdges(dataset: Datasets, threshold: int):
        if DataProcessor.__checkEdgeFile(dataset, threshold):
            frame = pd.read_csv(dataset.getEdgePath(threshold), header=None)
            frame.columns = ["node1", "node2"]
            return frame

    # HELPER

    @staticmethod
    def __checkEdgeFile(dataset: Datasets, threshold: int):
        if not exists(dataset.getEdgePath(threshold)):
            print(f"No edge file (edges.csv) found for {dataset.name}")
            return False
        return True

    # Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    @staticmethod
    def __processBankData() -> pd.DataFrame:
        dataset = Datasets.BANK_CLIENTS
        frame = pd.read_csv("../data/banks/raw/bank-full.csv", sep=";")

        for column in ["contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]:
            frame = frame.drop(column, axis=1)

        dataset.createDirectoryIfNotExists()

        frame.to_csv(dataset.getFeaturePath(), index_label="id")
        return frame

    # Source: https://archive.ics.uci.edu/ml/datasets/Adult
    @staticmethod
    def __processAdultData() -> pd.DataFrame:
        dataset = Datasets.ADULTS
        frame = pd.read_csv("../data/adults/raw/adult.data", header=None)
        frame.columns = ["age", "workclass", "finalWeight", "education", "education-num", "marital-status",
                         "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                         "native-country", "income"]

        dataset.createDirectoryIfNotExists()

        frame.to_csv(dataset.getFeaturePath(), index_label="id")
        return frame
