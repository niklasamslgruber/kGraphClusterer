import math

from dataHandler.datasets import Datasets
from engines.anonymizationType import AnonymizationType
from engines.distanceEngine import DistanceEngine
from engines.gilEngine import GILEngine
from engines.informationLossEngine import InformationLossEngine
from models.cluster import Cluster
from models.graph import Graph
from models.node import Node
from models.partition import Partition
import itertools
from tqdm import tqdm
import copy


class AnonymizationEngine:
    graph: Graph
    graph_nodes: [Node]
    alpha: float
    beta: float
    k: int
    dataset: Datasets
    type: AnonymizationType

    def __init__(self, graph: Graph, alpha: float, beta: float, k: int, dataset: Datasets, type: AnonymizationType):
        self.graph = graph
        self.graph_nodes = graph.nodes.copy()
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.dataset = dataset
        self.type = type

    def anonymize(self):
        S: {int: Cluster} = {}
        final_clusters: [Cluster] = []
        i: int = 1
        with tqdm(total=len(self.graph_nodes)) as pbar:
            while len(self.graph_nodes) != 0:
                old_length = len(self.graph_nodes)
                x_seed = self._getMaxDegreeNode()

                if i in S.keys():
                    S[i].nodes.append(x_seed)
                else:
                    cluster = Cluster(nodes=[x_seed])
                    cluster.id = i
                    x_seed.cluster_id = cluster.id
                    S[i] = cluster

                self.graph_nodes.remove(x_seed)

                while len(S[i].nodes) < self.k and len(self.graph_nodes) != 0:
                    X_star: Node
                    match self.type:
                        case AnonymizationType.SaNGreeA:
                            X_star = self._getArgminNode(self.alpha, self.beta, S[i])[1]
                        case AnonymizationType.DISCERNIBILITY:
                            engine = InformationLossEngine(self.alpha, self.beta, self.k, self.dataset, self.graph)
                            X_star, metric = engine.getDiscernibilityMetric(copy.copy(self.graph_nodes), copy.copy(S[i]))
                        case AnonymizationType.PRECISION:
                            engine = InformationLossEngine(self.alpha, self.beta, self.k, self.dataset, self.graph)
                            X_star, metric = engine.getPrecision(copy.copy(self.graph_nodes), copy.copy(S[i]))
                        case AnonymizationType.CLASSIFICATION_METRIC:
                            engine = InformationLossEngine(self.alpha, self.beta, self.k, self.dataset, self.graph)
                            X_star, metric = engine.getClassificationMetric(copy.copy(self.graph_nodes), copy.copy(S[i]))
                        case AnonymizationType.NORMALIZED_CERTAINTY_PENALTY:
                            engine = InformationLossEngine(self.alpha, self.beta, self.k, self.dataset, self.graph)
                            X_star, metric = engine.getNormalizedCertaintyPenalty(copy.copy(self.graph_nodes), copy.copy(S[i]))

                    S[i].nodes.append(X_star)
                    self.graph_nodes.remove(X_star)

                if len(S[i].nodes) < self.k:
                    value = S[i]
                    del S[i]
                    self.__disperseCluster(S, value)
                else:
                    cluster = Cluster(S[i].nodes)
                    cluster.id = i

                    for node in cluster.nodes:
                        node.cluster_id = cluster.id
                    final_clusters.append(cluster)
                    i += 1
                pbar.update(old_length - len(self.graph_nodes))

        return Partition(final_clusters)

    # SaNGreeA

    def __disperseCluster(self, partition_dict, cluster):
        partition = Partition(list(map(lambda key: partition_dict[key], partition_dict.keys())))
        for node in cluster.nodes:
            best_cluster = self.__findBestCluster(node, partition)[1]
            for key in partition_dict.keys():
                if partition_dict[key] == best_cluster:
                    node.cluster_id = key
                    partition_dict[key].nodes.append(node)

    def __findBestCluster(self, input_node, partition):
        min_loss = (math.inf, None)

        for cluster in partition.clusters:
            for node in cluster.nodes:
                new_cluster = Cluster(cluster.nodes.copy())
                new_cluster.nodes.append(node)
                s1 = new_cluster

                new_graph = self.graph
                ids = list(map(lambda node_item: node_item.id, new_cluster.nodes))
                untouched_nodes = list(filter(lambda node_item: node_item.id not in ids, new_graph.nodes))

                merged_cluster = GILEngine(self.graph, Partition([]), self.dataset).mergeNodes(s1)

                new_node = Node()
                new_node.id = node.id
                new_node.relations = list(
                    itertools.chain.from_iterable(
                        list(map(lambda node_item: node_item.relations, new_cluster.nodes))))
                new_node.degree = len(new_node.relations)
                new_node.value = merged_cluster

                untouched_nodes.append(new_node)

                new_graph.nodes = untouched_nodes

                new_engine = GILEngine(new_graph, Partition([new_cluster]), self.dataset)
                gil_value = self.alpha * new_engine.getNGIL()

                metric = gil_value + self.beta * DistanceEngine(self.graph).getNodeClusterDistance(cluster, input_node)

                if metric < min_loss[0]:
                    min_loss = [metric, cluster]

        return min_loss

    def _getArgminNode(self, alpha, beta, cluster):
        min_loss = (math.inf, None)

        for node in self.graph_nodes:
            gil_value = 0
            if alpha > 0:
                new_cluster = Cluster(cluster.nodes.copy())
                new_cluster.nodes.append(node)
                s1 = new_cluster

                new_graph = self.graph
                ids = list(map(lambda node_item: node_item.id, new_cluster.nodes))
                untouched_nodes = list(filter(lambda node_item: node_item.id not in ids, new_graph.nodes))

                merged_cluster = GILEngine(self.graph, Partition([]), self.dataset).mergeNodes(s1)

                new_node = Node()
                new_node.id = node.id
                new_node.relations = list(
                    itertools.chain.from_iterable(list(map(lambda node_item: node_item.relations, new_cluster.nodes))))
                new_node.degree = len(new_node.relations)
                new_node.value = merged_cluster

                untouched_nodes.append(new_node)

                new_graph.nodes = untouched_nodes

                new_engine = GILEngine(new_graph, Partition([new_cluster]), self.dataset)
                gil_value = alpha * new_engine.getNGIL()

            metric = gil_value + beta * DistanceEngine(self.graph).getNodeClusterDistance(cluster, node)

            if metric < min_loss[0]:
                min_loss = [metric, node]

        return min_loss

    def _getMaxDegreeNode(self):
        degrees = list(map(lambda y: y.degree, self.graph_nodes))

        return list(filter(lambda x: x.degree == max(degrees), self.graph_nodes))[0]
