import math
from engines.distanceEngine import DistanceEngine
from models.cluster import Cluster
from models.graph import Graph
from models.node import Node
import copy


class InformationLossEngine:

    alpha: float
    beta: float
    k: int

    def __init__(self, alpha: float, beta: float, k: int):
        self.alpha = alpha
        self.beta = beta
        self.k = k

    # http://www.tdp.cat/issues11/tdp.a169a14.pdf

    def getDiscernibilityMetric(self, graph: Graph, graph_nodes: [Node], S_original: {int: Cluster}) -> (Node, int):
        S = copy.deepcopy(S_original)
        optimal_case: (Node, int, int) = (None, math.inf, math.inf)
        for node in graph_nodes.copy():
            S = copy.deepcopy(S_original)

            for index in S:
                if len(S[index].nodes) >= self.k:
                    continue
                S = copy.deepcopy(S_original)
                S[index].nodes.append(node)
                clusters = list(map(lambda key: Cluster(S[key].nodes), S))
                all_cluster = clusters

                disc_metric = 0
                for cluster in all_cluster:
                    if len(cluster.nodes) < self.k:
                        disc_metric += self.alpha * len(graph.nodes) * len(cluster.nodes)
                    else:
                        disc_metric += self.alpha * math.pow(len(cluster.nodes), 2)

                    disc_metric += self.beta * DistanceEngine(graph).getNodeClusterDistance(cluster, node)

                if disc_metric < optimal_case[1]:
                    optimal_case = (node, disc_metric, index)

        return optimal_case

    # Result is equal to other DiscernibilityMetric if < is used above for overriding optimal_case, if <= is used the results vary
    def getDiscernibilityMetricForCurrentClusterOnly(self, graph: Graph, graph_nodes: [Node], S_original: Cluster) -> (Node, int):
        optimal_case: (Node, int) = (None, math.inf)
        for node in graph_nodes.copy():
            S = copy.deepcopy(S_original)
            S.nodes.append(node)
            cluster = Cluster(S.nodes)

            disc_metric = 0

            if len(cluster.nodes) < self.k:
                disc_metric += self.alpha * len(graph.nodes) * len(cluster.nodes)
            else:
                disc_metric += self.alpha * math.pow(len(cluster.nodes), 2)

            disc_metric += self.beta * DistanceEngine(graph).getNodeClusterDistance(cluster, node)

            if disc_metric < optimal_case[1]:
                optimal_case = (node, disc_metric)

        return optimal_case

    # def getPrecision:
    #     return 2
    #
    # def getNormalizedAverageEquivalenceClassSizeMetric:
    #     return 2.5
    #
    # def getClassificationMetric:
    #     return 3
    #
    # def getNormalizedCertaintyPenalty:
    #     return 4
    #
    #
    # def getEntropy:
    #     return 5

    # def
