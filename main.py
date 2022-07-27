import pandas as pd
from gilengine import GILEngine
from graph import Graph
from partition import Partition

if __name__ == '__main__':
    print("Starting Clusterer...")

    edges: pd.DataFrame = pd.read_csv("data/rawData/edges.csv", names=["node1", "node2"], header=None)
    features: pd.DataFrame = pd.read_csv("data/rawData/features.csv", index_col="node")

    numerical_identifiers: [str] = ["age"]
    categorical_identifiers: [str] = ["zip", "gender"]
    graph = Graph.create(edges, features, numerical_identifiers, categorical_identifiers)

    partition1 = Partition.create(graph, [[4, 7, 8], [1, 2, 3], [5, 6, 9]])
    partition2 = Partition.create(graph, [[4, 5, 6], [1, 2, 3], [7, 8, 9]])

    for (index, partition) in enumerate([partition1, partition2]):
        engine = GILEngine(graph, partition)
        print(f"Partition {index + 1}: \n\tNGIL: {engine.getNGIL()}\n\tGIL: {engine.getGraphGIL()}")
