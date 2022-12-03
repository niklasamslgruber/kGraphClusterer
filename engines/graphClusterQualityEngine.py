import math
import networkx as nx
from dataHandler.datasets import Datasets
from engines.distanceEngine import DistanceEngine
from models.cluster import Cluster
from models.graph import Graph
from models.node import Node
import copy


class GraphClusterQualityEngine:
    alpha: float
    beta: float
    k: int
    dataset: Datasets
    graph: Graph

    def __init__(self, alpha: float, beta: float, k: int, dataset: Datasets, graph: Graph):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.dataset = dataset
        self.graph = graph

    def getModularity(self, graph_nodes: [Node], clusters: [Cluster], index: int):
        optimal_case: (Node, int) = (None, -math.inf)
        for node in graph_nodes.copy():
            cluster_copy = copy.deepcopy(clusters)
            cluster_copy[index].nodes.append(node)

            modularity = 0

            for key in cluster_copy:
                a_sum = 0
                e_sum = 0
                cluster = cluster_copy[key]

                m = cluster.getNumberOfEdges()

                for cluster_node in cluster.nodes:
                    x = list(filter(lambda y: y not in cluster.getIds(), cluster_node.relations))
                    a_sum += len(x)

                    for relation in cluster_node.relations:
                        e_sum += 1 if relation in cluster.getIds() else 0

                a_sum = (1 / (2 * m)) * a_sum
                e_sum = (1 / (2 * m)) * e_sum

                modularity += e_sum - pow(a_sum, 2)

            modularity = self.alpha * modularity
            modularity -= self.beta * DistanceEngine(self.graph).getNodeClusterDistance(cluster_copy[index], node)

            if modularity > optimal_case[1]:
                optimal_case = (node, modularity)

        return optimal_case

    def getSilhouette(self, graph_nodes: [Node], clusters: [Cluster], index: int):
        optimal_case: (Node, int) = (None, -math.inf)
        for node in graph_nodes.copy():
            cluster_copy = copy.deepcopy(clusters)
            cluster_copy[index].nodes.append(node)

            G1 = nx.Graph()
            for key in cluster_copy:
                for cluster_node in cluster_copy[key].nodes:
                    G1.add_node(cluster_node.id)

                    for relation in cluster_node.relations:
                        G1.add_node(relation)
                        G1.add_edge(cluster_node.id, relation)

            cluster = cluster_copy[index]
            s_value = 0

            for cluster_node in cluster.nodes:
                a = 0
                b = 0
                a_counter = 0
                b_counter = 0
                for target_node in cluster.nodes:
                    if cluster_node != target_node:
                        try:
                            a += nx.shortest_path_length(G1, cluster_node.id, target_node.id)
                            a_counter += 1
                        except nx.NetworkXNoPath:
                            continue

                if index > 1:
                    prev_cluster = cluster_copy[index - 1]

                    for target_node in prev_cluster.nodes:
                        try:
                            b += nx.shortest_path_length(G1, cluster_node.id, target_node.id)
                            b_counter += 1
                        except nx.NetworkXNoPath:
                            continue

                if a_counter > 0:
                    a_avg = a / a_counter
                else:
                    a_avg = 0
                if b_counter > 0:
                    b_avg = b / b_counter
                else:
                    b_avg = 0
                s_value += (b_avg - a_avg) - max(a_avg, b_avg)

            silhouette = self.alpha * (s_value / len(clusters))
            silhouette -= self.beta * DistanceEngine(self.graph).getNodeClusterDistance(cluster, node)

            if silhouette > optimal_case[1]:
                optimal_case = (node, silhouette)

        return optimal_case

    def getGraphPerformance(self, graph_nodes: [Node], clusters: [Cluster], index: int):
        optimal_case: (Node, int) = (None, -math.inf)
        for node in graph_nodes.copy():
            cluster_copy = copy.deepcopy(clusters)
            cluster_copy[index].nodes.append(node)
            cluster = cluster_copy[index]

            all_nodes = list(map(lambda t: len(clusters[t].nodes), clusters))
            total_nodes = sum(all_nodes)

            f_C = cluster.getNumberOfEdges()
            number_non_existent_edges = 0

            for key in cluster_copy:
                target_cluster = cluster_copy[key]
                if cluster == target_cluster:
                    continue

                for cluster_node in cluster.nodes:
                    target_ids = set(target_cluster.getIds())
                    for relation in cluster_node.relations:
                        if relation in target_ids:
                            target_ids.remove(relation)

                    number_non_existent_edges += len(target_ids)

            performance = self.alpha * ((f_C + number_non_existent_edges) / 0.5 * total_nodes * (total_nodes - 1))
            performance -= self.beta * DistanceEngine(self.graph).getNodeClusterDistance(cluster, node)

            if performance > optimal_case[1]:
                optimal_case = (node, performance)

        return optimal_case

    # Source: https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf
    def getPurity(self, graph_nodes: [Node], S_original: Cluster):
        optimal_case: (Node, int) = (None, -math.inf)
        majority = S_original.getMajority(self.dataset)

        for node in graph_nodes.copy():
            cluster = copy.deepcopy(S_original)

            cluster.nodes.append(node)

            majority_counter = 0
            for cluster_node in cluster.nodes:
                for identifier in self.dataset.getNumericalIdentifiers() + self.dataset.getCategoricalIdentifiers():
                    if cluster_node.value[identifier] == majority[identifier]:
                        majority_counter += 1

            purity = self.alpha * (majority_counter / (len(cluster.nodes) * (
                len(self.dataset.getNumericalIdentifiers() + self.dataset.getCategoricalIdentifiers()))))
            purity -= self.beta * DistanceEngine(self.graph).getNodeClusterDistance(cluster, node)

            if purity > optimal_case[1]:
                optimal_case = (node, purity)

        return optimal_case
