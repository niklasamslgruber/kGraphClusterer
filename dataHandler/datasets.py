from os.path import exists
from os import mkdir
from enum import Enum, unique
from constants import ROOT_DIR


@unique
class Datasets(Enum):
    ADULTS = f"{ROOT_DIR}/data/adults/adults.csv"
    BANK_CLIENTS = f"{ROOT_DIR}/data/banks/banks.csv"
    SAMPLE = f"{ROOT_DIR}/data/sample/sample.csv"

    def getDirectory(self):
        return "/".join(self.value.split("/")[:-1])

    def createDirectoryIfNotExists(self):
        if not exists(self.getDirectory()):
            mkdir(self.getDirectory())

    def getEdgePath(self):
        return f"{self.getDirectory()}/edges.csv"

    def getGeneralizationTree(self, attribute: str):
        return f"{self.getDirectory()}/trees/{attribute}_generalization_tree.json"

    def getResultsPath(self):
        return f"{self.getDirectory()}/results.csv"

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
