import pandas as pd
from engines.anonymizationEngine import AnonymizationEngine
from dataHandler.dataProcessor import DataProcessor
from dataHandler.datasets import Datasets
from engines.gilEngine import GILEngine
from engines.graphEvaluationEngine import GraphEvaluationEngine
from dataHandler.graphGenerator import GraphGenerator
from models.graph import Graph
import copy

if __name__ == '__main__':
    print("Starting Clusterer...\n")

    edges: pd.DataFrame = DataProcessor.loadEdges(Datasets.SAMPLE)
    features: pd.DataFrame = DataProcessor.loadFeatures(Datasets.SAMPLE)

    numerical_identifiers: [str] = ["age"]
    categorical_identifiers: [str] = ["zip", "gender"]
    graph = Graph.create(edges, features, numerical_identifiers, categorical_identifiers)

    partition = AnonymizationEngine(copy.copy(graph), 1, 0, 3).anonymize()
    print("Clusters:")
    for cluster in partition.clusters:
        print("\t", cluster.getIds())

    nsil = GraphEvaluationEngine().getNSIL(partition, graph)
    ngil = GILEngine(graph, partition).getNGIL()
    print("NSIL:", nsil)
    print("NGIL:", ngil)


    # DataProcessor.loadData(DataProcessor.Dataset.BANK_CLIENTS)
    # DataProcessor.loadData(DataProcessor.Dataset.ADULTS)

    GraphGenerator.generateRandomEdges(Datasets.ADULTS, num=100)
    GraphGenerator.generateRandomEdges(Datasets.BANK_CLIENTS, num=100, limit=1000)

