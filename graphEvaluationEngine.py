from models.cluster import Cluster
from models.partition import Partition
import math


class GraphEvaluationEngine:

    # Metrics

    def getSIL(self, partition):
        total_inter_loss = 0
        total_intra_loss = 0

        for interLoss in self._getInterLossesForPartition(partition):
            total_inter_loss += interLoss[2]

        for intraLoss in self._getIntraLossesForPartition(partition):
            total_intra_loss += intraLoss[1]

        return total_inter_loss + total_intra_loss

    def getNSIL(self, partition, graph):
        total_nodes = len(graph.nodes)

        return self.getSIL(partition) / (total_nodes * (total_nodes - 1) / 4)

    # Intra-cluster structural information loss

    def _getIntraLossesForPartition(self, partition: Partition):
        values: [([int], float)] = []
        for cluster in partition.clusters:
            values.append((cluster.getIds(), self._getIntraLossForCluster(cluster)))

        return values

    def _getIntraLossForCluster(self, cluster: Cluster):
        numberOfNodes = len(cluster.nodes)
        numberOfEdges = self._getNumberOfEdgesInCluster(cluster)

        return 2 * numberOfEdges * (1 - (numberOfEdges / self._getNOfM(numberOfNodes, 2)))

    # Inter-cluster structural information loss

    def _getInterLossesForPartition(self, partition: Partition):
        values: [([int], [int], float)] = []
        for (index, cluster) in enumerate(partition.clusters):
            for item in range(index + 1, len(cluster.nodes)):
                cluster2 = partition.clusters[item]
                values.append((cluster.getIds(), cluster2.getIds(), self._getInterLossForCluster(cluster, cluster2)))

        return values

    def _getInterLossForCluster(self, cluster: Cluster, cluster2: Cluster):
        totalNumberOfEdges = 0
        ids = cluster2.getIds()

        for node in cluster.nodes:
            for item in node.relations:
                if item in ids:
                    totalNumberOfEdges += 1

        return 2 * totalNumberOfEdges * (1 - (totalNumberOfEdges / (len(cluster.nodes) * len(cluster2.nodes))))

    # - HELPER

    # n = Top, m = Bottom
    def _getNOfM(self, n: int, m: int):
        return math.factorial(n) / (math.factorial(n - m) * math.factorial(m))

    def _getNumberOfEdgesInCluster(self, cluster: Cluster):
        numberOfEdges = 0
        ids = cluster.getIds()
        for node in cluster.nodes:
            for item in node.relations:
                if item in ids:
                    numberOfEdges += 1

        numberOfEdges = numberOfEdges / 2
        return numberOfEdges
