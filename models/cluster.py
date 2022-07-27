from models.node import Node
from models.graph import Graph


class Cluster:
    nodes: [Node] = []

    def __init__(self, nodes: [Node]):
        self.nodes = nodes

    @staticmethod
    def create(graph: Graph, ids: [int]):
        return Cluster(list(filter(lambda node: node.id in ids, graph.nodes)))
