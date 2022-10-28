from dataHandler.datasets import Datasets
from models.node import Node
from models.graph import Graph


class Cluster:
    id: str
    nodes: [Node] = []

    def __init__(self, nodes: [Node]):
        self.nodes = nodes

    @staticmethod
    def create(graph: Graph, ids: [int]):
        return Cluster(list(filter(lambda node: node.id in ids, graph.nodes)))

    def getIds(self):
        return list(map(lambda node: node.id, self.nodes))

    def getNumberOfEdges(self):
        edge_count = 0
        for node in self.nodes:
            edge_count += len(node.relations)

        return edge_count

    def getMajority(self, dataset: Datasets):
        majority: {str: str} = {}
        values = list(map(lambda node: node.value, self.nodes))
        for identifier in dataset.getNumericalIdentifiers() + dataset.getCategoricalIdentifiers():
            valueX = list(map(lambda value: value[identifier], values))
            majority[identifier] = max(valueX, key=valueX.count)

        return majority
