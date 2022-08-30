from os.path import exists
from os import mkdir
from enum import Enum, unique


@unique
class Datasets(Enum):
    ADULTS = "data/adults/adults.csv"
    BANK_CLIENTS = "data/banks/banks.csv"
    SAMPLE = "data/sample/sample.csv"

    def getDirectory(self):
        return "/".join(self.value.split("/")[:-1])

    def createDirectoryIfNotExists(self):
        if not exists(self.getDirectory()):
            mkdir(self.getDirectory())

    def getEdgePath(self):
        return f"{self.getDirectory()}/edges.csv"

    def getGeneralizationTree(self, attribute: str):
        return f"{self.getDirectory()}/trees/{attribute}_generalization_tree.json"
