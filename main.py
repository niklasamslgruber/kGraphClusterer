import pandas as pd
from anonymizationEngine import AnonymizationEngine
from informationLossEngine import InformationLossEngine
from models.graph import Graph
import copy

if __name__ == '__main__':
    print("Starting Clusterer...")

    edges: pd.DataFrame = pd.read_csv("data/rawData/edges.csv", names=["node1", "node2"], header=None)
    features: pd.DataFrame = pd.read_csv("data/rawData/features.csv", index_col="node")

    numerical_identifiers: [str] = ["age"]
    categorical_identifiers: [str] = ["zip", "gender"]
    graph = Graph.create(edges, features, numerical_identifiers, categorical_identifiers)

    partition = AnonymizationEngine(copy.copy(graph), 1, 0, 3).anonymize()
    nsil = InformationLossEngine().getNSIL(partition, graph)

    partition2 = AnonymizationEngine(copy.copy(graph), 0, 1, 3).anonymize()
    nsil2 = InformationLossEngine().getNSIL(partition2, graph)
    print(nsil2)
