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
