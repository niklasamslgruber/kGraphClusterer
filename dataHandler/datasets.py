from os.path import exists
from os import mkdir
from enum import Enum, unique
from constants import ROOT_DIR


@unique
class Datasets(Enum):
    ADULTS = f"{ROOT_DIR}/data/adults/adults.csv"
    BANK_CLIENTS = f"{ROOT_DIR}/data/banks/banks.csv"
    SAMPLE = f"{ROOT_DIR}/data/sample/sample.csv"

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

    def getDirectory(self):
        return "/".join(self.value.split("/")[:-1])

    def createDirectoryIfNotExists(self):
        if not exists(self.getDirectory()):
            mkdir(self.getDirectory())

    def getEdgeDirectory(self):
        path = f"{self.getDirectory()}/edges"
        if not exists(path):
            mkdir(path)
        return path

    def getEdgePath(self, threshold: int):
        if self == Datasets.SAMPLE:
            return f"{self.getEdgeDirectory()}/edges.csv"
        return f"{self.getEdgeDirectory()}/edges_{threshold}.csv"

    def getAssociationDirectory(self):
        path = f"{self.getDirectory()}/associations"
        if not exists(path):
            mkdir(path)
        return path

    def getAssociationPath(self, threshold: int):
        if self == Datasets.SAMPLE:
            return f"{self.getAssociationDirectory()}/associations.csv"
        return f"{self.getAssociationDirectory()}/associations_{threshold}.csv"

    def getGeneralizationTree(self, attribute: str):
        return f"{self.getDirectory()}/trees/{attribute}_generalization_tree.json"

    def getResultsPath(self, old: bool = False):
        return f"{self.getDirectory()}/{'results_random' if old else 'results'}.csv"

    def getImagePath(self):
        path = f"{self.getDirectory()}/images"
        if not exists(path):
            mkdir(path)
        return path

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
