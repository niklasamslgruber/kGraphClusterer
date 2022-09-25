import pandas as pd

from dataHandler.datasets import Datasets
from models.node import Node


class Graph:
    nodes: [Node] = []
    numerical_identifiers: [str] = []
    categorical_identifiers: [str] = []
    features: pd.DataFrame

    @staticmethod
    def create(edges: pd.DataFrame, features: pd.DataFrame, dataset: Datasets):
        graph = Graph()
        graph.nodes = []
        graph.features = features

        assert "node1" in edges.columns and "node2" in edges.columns, f"Edges need to have the headers 'node1' and 'node2', but found only {edges.columns.values}"
        for index, row in features.iterrows():
            node = Node()
            node.id = row["transactionID"]
            node.relations = edges[edges["node1"] == node.id]["node2"].to_list()
            node.degree = len(node.relations)
            node.value = features.loc[index].to_dict()
            graph.nodes.append(node)

        graph.numerical_identifiers = dataset.getNumericalIdentifiers()
        graph.categorical_identifiers = dataset.getCategoricalIdentifiers()

        assert len(graph.nodes) == features.index.size, f"Number of features ({features.index.size}) does not match number of nodes ({len(graph.nodes)}"
        return graph

    def getIds(self):
        return list(map(lambda node: node.id, self.nodes))
