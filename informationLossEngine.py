import math

from models.cluster import Cluster
from models.graph import Graph
from models.node import Node
import copy

class InformationLossEngine:

    # http://www.tdp.cat/issues11/tdp.a169a14.pdf
    def getDiscernibilityMetric(self, graph: Graph, graph_nodes: [Node], S_original: {int: Cluster}, k: int) -> (Node, int):
        S = copy.deepcopy(S_original)
        optimal_case: (Node, int) = (None, math.inf)
        for node in graph_nodes.copy():
            S = copy.deepcopy(S_original)
            # print(f"Analyze node {node.id}")
            untouchedNodes = list(filter(lambda x: x.id != node.id, graph_nodes))

            for index in S:
                S = copy.deepcopy(S_original)
                # print(f"\nAttach to cluster {index}")
                S[index].nodes.append(node)
                clusters = list(map(lambda key: Cluster(S[key].nodes), S))
                all_cluster = clusters

                disc_metric = 0
                for cluster in all_cluster:
                    # print(f"Cluster Nodes {list(map(lambda y: y.id, cluster.nodes))}")
                    if len(cluster.nodes) < k:
                        disc_metric += len(graph.nodes) * len(cluster.nodes)
                    else:
                        disc_metric += math.pow(len(cluster.nodes), 2)
                # print(f"Metric {disc_metric}")
                if disc_metric < optimal_case[1]:
                    optimal_case = (node, disc_metric)

        return optimal_case


                # for x in clusters:
                #     print(f"Cluster {x.getIds()}")

        # S = S_original




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