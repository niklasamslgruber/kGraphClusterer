import pandas as pd

from dataHandler.graphGenerator import GraphGenerator
from engines.anonymizationEngine import AnonymizationEngine
from dataHandler.dataProcessor import DataProcessor
from dataHandler.datasets import Datasets
from engines.gilEngine import GILEngine
from engines.graphEvaluationEngine import GraphEvaluationEngine
from models.graph import Graph
import copy


def start(dataset: Datasets, alpha: int, beta: int, k: int, limit: int = -1):
    edges: pd.DataFrame = DataProcessor.loadEdges(dataset)
    features: pd.DataFrame = DataProcessor.loadFeatures(dataset)

    if limit != -1:
        features = features.head(limit)

    graph = Graph.create(edges, features, dataset)

    partition = AnonymizationEngine(copy.copy(graph), alpha, beta, k, dataset).anonymize()
    print("\n-----")
    print(f"Generated Clusters for dataset {dataset.name} (k = {k}, alpha = {alpha}, beta = {beta}):")
    for (index, cluster) in enumerate(partition.clusters):
        print(f"\tCluster {index + 1}:", cluster.getIds())

    print("\nStatistics:")
    nsil = GraphEvaluationEngine(partition, graph).getNSIL()
    ngil = GILEngine(graph, partition, dataset).getNGIL()
    print("\tNSIL:", nsil)
    print("\tNGIL:", ngil)


if __name__ == '__main__':
    print("Starting Clusterer...\n")

    limit = 100

    start(Datasets.ADULTS, 1, 0, 3, limit=limit)
    # start(Datasets.ADULTS, 0, 1, 3, limit=limit)

    # DataProcessor.getUniqueValues(Datasets.ADULTS, [])

    # GraphGenerator.generateRandomEdges(Datasets.ADULTS, num=500, limit=limit, force=True)
    # GraphGenerator.generateRandomEdges(Datasets.BANK_CLIENTS, num=500, limit=limit, force=True)
