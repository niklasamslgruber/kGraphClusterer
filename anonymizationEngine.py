import copy

from distanceEngine import DistanceEngine
from gilEngine import GILEngine
from informationLossEngine import InformationLossEngine
from models.cluster import Cluster
from models.graph import Graph
from models.node import Node
from models.partition import Partition
import itertools


class AnonymizationEngine:
    graph: Graph
    graph_nodes: [Node]
    alpha: float
    beta: float
    k: int

    def __init__(self, graph: Graph, alpha: float, beta: float, k: int):
        self.graph = graph
        self.graph_nodes = graph.nodes.copy()
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def anonymize(self):
        S: {int: Cluster} = {}
        final_clusters: [Cluster] = []
        i: int = 1

        while len(self.graph_nodes) != 0:
            x_seed = self._getMaxDegreeNode()

            if i in S.keys():
                S[i].nodes.append(x_seed)
            else:
                S[i] = Cluster(nodes=[x_seed])

            self.graph_nodes.remove(x_seed)

            while len(S[i].nodes) < self.k and len(self.graph_nodes) != 0:
                X_star = self._getArgminNode(self.alpha, self.beta, S[i])[1]
                index = i

                # X_star, metric, index = InformationLossEngine().getDiscernibilityMetric(self.graph, copy.copy(self.graph_nodes), copy.copy(S), self.k)
                # X_star, metric = InformationLossEngine().getDiscernibilityMetricForCurrentClusterOnly(self.graph, copy.copy(self.graph_nodes), copy.copy(S[i]), self.k)

                S[index].nodes.append(X_star)
                self.graph_nodes.remove(X_star)

            if len(S[i].nodes) < self.k:
                # TODO: Integrate disperse cluster
                print('Disperse Cluster')
            else:
                final_clusters.append(Cluster(S[i].nodes))
                i += 1

        return Partition(final_clusters)

    def _getArgminNode(self, alpha, beta, cluster):
        min_loss = (99999, None)

        for node in self.graph_nodes:
            gil_value = 0
            if alpha > 0:
                new_cluster = Cluster(cluster.nodes.copy())
                new_cluster.nodes.append(node)
                s1 = new_cluster

                new_graph = self.graph
                ids = list(map(lambda node_item: node_item.id, new_cluster.nodes))
                untouched_nodes = list(filter(lambda node_item: node_item.id not in ids, new_graph.nodes))

                merged_cluster = GILEngine(self.graph, Partition([])).mergeNodes(s1)

                new_node = Node()
                new_node.id = node.id
                new_node.relations = list(
                    itertools.chain.from_iterable(list(map(lambda node_item: node_item.relations, new_cluster.nodes))))
                new_node.degree = len(new_node.relations)
                new_node.value = merged_cluster

                untouched_nodes.append(new_node)

                new_graph.nodes = untouched_nodes

                new_engine = GILEngine(new_graph, Partition([new_cluster]))
                gil_value = alpha * new_engine.getNGIL()

            metric = gil_value + beta * DistanceEngine(self.graph).getNodeClusterDistance(cluster, node)

            if metric < min_loss[0]:
                min_loss = [metric, node]

        return min_loss

    def _getMaxDegreeNode(self):
        degrees = list(map(lambda y: y.degree, self.graph_nodes))

        return list(filter(lambda x: x.degree == max(degrees), self.graph_nodes))[0]
