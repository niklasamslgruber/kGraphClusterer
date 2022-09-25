import json
import math
from dataHandler.datasets import Datasets
from engines.distanceEngine import DistanceEngine
from engines.gilEngine import GILEngine
from models.cluster import Cluster
from models.graph import Graph
from models.node import Node
import copy

from models.partition import Partition


class InformationLossEngine:
    alpha: float
    beta: float
    k: int
    dataset: Datasets

    def __init__(self, alpha: float, beta: float, k: int, dataset: Datasets):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.dataset = dataset

    # http://www.tdp.cat/issues11/tdp.a169a14.pdf
    def getDiscernibilityMetric(self, graph: Graph, graph_nodes: [Node], S_original: Cluster) -> (
            Node, int):
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

    # Source: https://dataprivacylab.org/dataprivacy/projects/kanonymity/kanonymity2.pdf
    def getPrecision(self, graph: Graph, graph_nodes: [Node], S_original: Cluster) -> (Node, int):
        dghs: {str: dict} = {}
        for categorical_attribute in self.dataset.getCategoricalIdentifiers():
            with open(self.dataset.getGeneralizationTree(categorical_attribute)) as json_file:
                data = json.load(json_file)
                dghs[categorical_attribute] = data

        numerical_dghs: {str: float} = {}
        for numerical_attribute in self.dataset.getNumericalIdentifiers():
            graph_values = list(map(lambda graph_node: graph_node.value[numerical_attribute], graph.nodes))
            numerical_dghs[numerical_attribute] = max(graph_values) - min(graph_values)

        optimal_case: (Node, int) = (None, 0)
        for node in graph_nodes.copy():
            S = copy.deepcopy(S_original)
            S.nodes.append(node)

            cluster = Cluster(S.nodes)
            generalized_values = GILEngine(graph, Partition([]), self.dataset).mergeNodes(cluster)
            sum_value = 0

            for categorical_attribute in self.dataset.getCategoricalIdentifiers():
                data = dghs[categorical_attribute]
                DGH = GILEngine.getTreeDepth(data)
                subtree = GILEngine.getSubHierarchyTree(data, generalized_values[categorical_attribute][0])

                h = GILEngine.getTreeDepth(subtree)
                if h != 0:
                    h -= 1

                sum_value += (h / DGH) * len(cluster.nodes)

            for numerical_attribute in self.dataset.getNumericalIdentifiers():
                new_value = generalized_values[numerical_attribute]

                DGH = numerical_dghs[numerical_attribute]
                h = max(new_value) - min(new_value)

                sum_value += (h / DGH) * len(cluster.nodes)

            disc_metric = self.alpha * (1 - (sum_value / ((len(self.dataset.getCategoricalIdentifiers()) + len(
                self.dataset.getNumericalIdentifiers())) * len(cluster.nodes))))

            # Subtract distance because distance is better if smaller but precision is better if bigger
            disc_metric -= self.beta * DistanceEngine(graph).getNodeClusterDistance(cluster, node)

            if disc_metric > optimal_case[1]:
                optimal_case = (node, disc_metric)

        return optimal_case

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
