from models.graph import Graph
from models.node import Node
from models.cluster import Cluster


class DistanceEngine:
    graph: Graph

    def __init__(self, graph: Graph):
        self.graph = graph

    # Distance between a node and a cluster (Definition 12)
    def getNodeClusterDistance(self, cluster: Cluster, node: Node):
        total_distance = 0
        for cluster_node in cluster.nodes:
            total_distance += self._getNodeDistance(cluster_node, node)

        return total_distance / len(cluster.nodes)

    # Distance between two nodes (Definition 11)
    def _getNodeDistance(self, node1: Node, node2: Node):
        b1 = list(map(lambda node: 1 if node.id in node1.relations else 0, self.graph.nodes))
        b2 = list(map(lambda node: 1 if node.id in node2.relations else 0, self.graph.nodes))

        cardinality = 0
        for (index, value) in enumerate(b1):
            new_index = index + 1
            if new_index != node1.id and new_index != node2.id and value != b2[index]:
                cardinality += 1

        if len(self.graph.nodes) == 2:
            return 0
        return cardinality / (len(self.graph.nodes) - 2)
