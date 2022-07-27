import json
from models.cluster import Cluster
from models.graph import Graph
from models.partition import Partition


class GILEngine:
    graph: Graph
    partition: Partition

    def __init__(self, graph: Graph, partition: Partition):
        self.graph = graph
        self.partition = partition

    # - CALCULATIONS

    # Normalized Generalization Information Loss for whole Graph (Definition 6)
    def getNGIL(self):
        return self.getGraphGIL() / (
                len(self.graph.nodes) * (len(self.graph.categorical_identifiers) + len(self.graph.numerical_identifiers)))

    # Total Generalization Information Loss for whole Graph (Definition 5)
    def getGraphGIL(self):
        total_loss = 0
        for partition in self.partition.clusters:
            total_loss += self._getClusterGIL(partition)

        return total_loss

    # Generalization Information Loss for Cluster (Definition 4)
    def _getClusterGIL(self, cluster: Cluster):
        clusterSize = len(cluster.nodes)
        numerical_sum = 0
        categorical_sum = 0

        # Numerical Identifiers
        for numerical_attribute in self.graph.numerical_identifiers:
            cluster_interval = list(map(lambda nodeItem: nodeItem.value[numerical_attribute], cluster.nodes))
            size_cluster = max(cluster_interval) - min(cluster_interval)

            total_interval = list(map(lambda nodeItem: nodeItem.value[numerical_attribute], self.graph.nodes))
            size_total = max(total_interval) - min(total_interval)
            numerical_sum += size_cluster / size_total

        # Categorical Identifiers
        for categorical_attribute in self.graph.categorical_identifiers:
            should_subtract = True
            with open(f'data/generalizationTrees/{categorical_attribute}_generalization_tree.json') as json_file:
                data = json.load(json_file)
                total_height = self._getTreeDepth(data) - 1

                generalized_keys = []
                values = list(set(map(lambda x: x.value[categorical_attribute], cluster.nodes)))
                if len(values) == 1:
                    generalized_sum_key = values[0]
                    should_subtract = False
                else:
                    for node in cluster.nodes:
                        generalized_key = self._getCommonAncestor(str(node.value[categorical_attribute]), data)
                        if generalized_key is not None:
                            generalized_keys.append(generalized_key)

                    generalized_sum_key = self._getGeneralization(data, generalized_keys)

                sub_generalization_tree = self._getSubHierarchyTree(data, generalized_sum_key)
                sub_height = self._getTreeDepth(sub_generalization_tree)
                if should_subtract is True:
                    sub_height -= 1

                categorical_sum += sub_height / total_height

        return clusterSize * (numerical_sum + categorical_sum)

    # - HELPER

    # Finds a common ancestor key for any data, i.e. the lowest possible key in the JSON for all values
    def _getCommonAncestor(self, searchKey: str, values: dict):
        data = values.copy()
        return_value = None

        if searchKey in data.keys():
            return searchKey

        for key, value in data.items():
            if isinstance(value, dict):
                if searchKey in data[key].keys():
                    return_value = key
                else:
                    value = self._getCommonAncestor(searchKey, data[key])
                    if value is None:
                        continue
                    else:
                        return_value = value

            elif isinstance(value, list):
                if str(searchKey) in value:
                    return_value = key
                else:
                    continue
            else:
                if value == searchKey or key == searchKey:
                    return_value = key
                else:
                    continue

        return return_value

    # Calculates the depth of the generalization tree - height(X)
    def _getTreeDepth(self, data: dict):
        if type(data) is dict and data:
            return 1 + max(self._getTreeDepth(data[a]) for a in data)
        if type(data) is list and data:
            return 1 + max(self._getTreeDepth(a) for a in data)
        return 0

    # Generates a subtree of the generalization tree for a given key - A(gen(cl)[Cj])
    def _getSubHierarchyTree(self, data: dict, searchKey: str):
        if searchKey not in data.keys():
            for key, value in data.items():
                if isinstance(value, dict):
                    return self._getSubHierarchyTree(data[key], searchKey)
                elif isinstance(value, list):
                    if searchKey in value:
                        return searchKey
                    else:
                        continue
                else:
                    if value == searchKey or key == searchKey:
                        return searchKey
                    else:
                        continue
        else:
            return data

    # Calculates the lowest possible generalization for all values - gen(cl)
    def _getGeneralization(self, data: dict, values: [str]):
        if len(set(values)) == 1:
            return list(set(values))[0]
        ancestors = []
        for item in values:
            ancestor = self._getCommonAncestor(item, data)
            if ancestor is not None:
                ancestors.append(ancestor)

        if len(set(ancestors)) == 1:
            return list(set(ancestors))[0]
        else:
            for ancestor in ancestors.copy():
                ancestors.remove(ancestor)
                ancestors.append(self._getCommonAncestor(ancestor, data))
                if len(set(ancestors)) == 1:
                    return list(set(ancestors))[0]
            return self._getGeneralization(data, ancestors)
