from models.cluster import Cluster


class Partition:
    clusters: [Cluster]

    def __init__(self, clusters: [Cluster]):
        self.clusters = clusters

    @staticmethod
    def create(graph, distribution: [[int]]):
        clusters: [Cluster] = []
        for identifiers in distribution:
            clusters.append(Cluster.create(graph, identifiers))

        return Partition(clusters)

    def getIds(self):
        return list(map(lambda cluster: cluster.getIds(), self.clusters))
