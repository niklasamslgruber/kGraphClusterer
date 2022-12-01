from datetime import datetime
from os.path import exists
from os import mkdir
from enum import Enum, unique
from constants import ROOT_DIR


@unique
class Datasets(Enum):
    ADULTS = "adults"
    BANK_CLIENTS = "banks"
    SAMPLE = "sample"

    @staticmethod
    def getCase(dataset: str):
        match dataset:
            case Datasets.ADULTS.name:
                return Datasets.ADULTS
            case Datasets.BANK_CLIENTS.name:
                return Datasets.BANK_CLIENTS
            case Datasets.SAMPLE.name:
                return Datasets.SAMPLE
            case _:
                return

    def createDirectoryIfNotExists(self):
        if not exists(self.__getDirectory()):
            mkdir(self.__getDirectory())

    def getFeaturePath(self):
        return f"{self.__getDirectory()}/features.csv"

    def getEdgePath(self, threshold: int):
        if self == Datasets.SAMPLE:
            return f"{self.__getEdgeDirectory()}/edges.csv"
        return f"{self.__getEdgeDirectory()}/edges_{threshold}.csv"

    def getAssociationPath(self, threshold: int):
        if self == Datasets.SAMPLE:
            return f"{self.__getAssociationDirectory()}/associations.csv"
        return f"{self.__getAssociationDirectory()}/associations_{threshold}.csv"

    def getGeneralizationTree(self, attribute: str):
        return f"{self.__getDirectory()}/trees/{attribute}_generalization_tree.json"

    def getResultsPath(self):
        return f"{self.__getDirectory()}/results.csv"

    def getImagePath(self):
        path = f"{self.__getDirectory()}/images"
        if not exists(path):
            mkdir(path)
        return path

    def getOutputPath(self):
        now = datetime.now().strftime("%d-%m-%Y_%H:%M")
        base_path = f"{self.__getDirectory()}/output"
        if not exists(base_path):
            mkdir(base_path)
        path = f"{base_path}/{now}/"
        if not exists(path):
            mkdir(path)
        return path

    # Identifiers

    def getCategoricalIdentifiers(self):
        match self:
            case Datasets.SAMPLE:
                return ["zip", "gender"]
            case Datasets.BANK_CLIENTS:
                return ["job", "marital", "education", "housing", "loan"]
            case Datasets.ADULTS:
                return ["workclass", "marital-status", "race", "sex", "native-country"]
            case _:
                return []

    def getNumericalIdentifiers(self):
        match self:
            case Datasets.SAMPLE:
                return ["age"]
            case Datasets.BANK_CLIENTS:
                return ["age", "balance"]
            case Datasets.ADULTS:
                return ["age"]
            case _:
                return []

    # Helper

    def __getDirectory(self):
        return "/".join(f"{ROOT_DIR}/data/{self.value}".split("/"))

    def __getEdgeDirectory(self):
        path = f"{self.__getDirectory()}/edges"
        if not exists(path):
            mkdir(path)
        return path

    def __getAssociationDirectory(self):
        path = f"{self.__getDirectory()}/associations"
        if not exists(path):
            mkdir(path)
        return path

