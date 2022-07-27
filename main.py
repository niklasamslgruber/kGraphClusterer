import pandas as pd

from anonymizationEngine import AnonymizationEngine
from distanceEngine import DistanceEngine
from gilEngine import GILEngine
from models.cluster import Cluster
from models.graph import Graph
from models.partition import Partition

if __name__ == '__main__':
    print("Starting Clusterer...")

    edges: pd.DataFrame = pd.read_csv("data/rawData/edges.csv", names=["node1", "node2"], header=None)
    features: pd.DataFrame = pd.read_csv("data/rawData/features.csv", index_col="node")

    numerical_identifiers: [str] = ["age"]
    categorical_identifiers: [str] = ["zip", "gender"]
    graph = Graph.create(edges, features, numerical_identifiers, categorical_identifiers)

    print(AnonymizationEngine(graph, 1, 0, 3).anonymize().getIds())
