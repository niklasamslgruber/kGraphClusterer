import pandas as pd
from os.path import exists

from dataHandler.datasets import Datasets


class ResultCollector:
    class Result:
        k: int
        alpha: float
        beta: float
        size: int
        edge_num: int
        ngil: float
        nsil: float
        num_of_clusters: int
        time: float
        method: str
        vertex_degree: int

        def __init__(self, k: int, alpha: float, beta: float, size: int, edge_num: int, ngil: float, nsil: float, num_of_clusters: int, time: float, method: str, vertex_degree: int):
            self.k = k
            self.alpha = alpha
            self.beta = beta
            self.size = size
            self.edge_num = edge_num
            self.ngil = ngil
            self.nsil = nsil
            self.num_of_clusters = num_of_clusters
            self.time = time
            self.method = method
            self.vertex_degree = vertex_degree

        def to_dict(self):
            return {
                "k": self.k,
                "alpha": self.alpha,
                "beta": self.beta,
                "size": self.size,
                "edge_num": self.edge_num,
                "ngil": self.ngil,
                "nsil": self.nsil,
                "num_of_clusters": self.num_of_clusters,
                "time": self.time,
                "method": self.method,
                "vertex_degree": self.vertex_degree
            }

    dataset: Datasets

    def __init__(self, dataset: Datasets):
        self.dataset = dataset

    def loadResults(self) -> pd.DataFrame:
        frame: pd.DataFrame
        if exists(self.dataset.getResultsPath()):
            frame = pd.read_csv(self.dataset.getResultsPath())
        else:
            frame = pd.DataFrame()

        return frame

    def saveResult(self, newResult: Result):
        existingResults = self.loadResults()

        newData = newResult.to_dict()

        if len(existingResults) == 0:
            existingResults = pd.DataFrame([newData])
        else:
            existingResults = pd.concat([existingResults, pd.DataFrame([newData])], ignore_index=True)

        existingResults.columns = newData.keys()
        existingResults.to_csv(self.dataset.getResultsPath(), index=False)
