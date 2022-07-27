import pandas as pd
import json

class Node:
    id: int
    degree: int
    relations: [int]
    value: {} = {}

    def __str__(self):
        return f'Id: {self.id}\n' \
               f'Degree: {self.degree}\n' \
               f'Relations: {list(map(lambda x: str(x), self.relations))}' \
               f'Value: {self.value}'


class Cluster:
    nodes: [Node]


class Clusterer:
    nodes: [Node] = []
    numerical_identifiers: [str] = ["age"]
    categorical_identifiers: [str] = ["zip", "gender"]
    partitions1: [[Node]] = [[]]
    partitions2: [[Node]] = [[]]

    def __init__(self):
        self.prepareData()
        self.calculateValues()

    def prepareData(self):
        edges = pd.read_csv("edges.csv", names=["node1", "node2"], header=None)
        features = pd.read_csv("features.csv", index_col="node")
        for index, row in features.iterrows():
            node = Node()
            node.id = index
            node.relations = edges[edges["node1"] == index]["node2"].to_list()
            node.degree = len(node.relations)
            node.value = features.loc[index].to_dict()
            self.nodes.append(node)

        assert len(self.nodes) == features.index.size

        self.partitions1 = [
            list(filter(lambda x: x.id in {4, 7, 8}, self.nodes)),
            list(filter(lambda x: x.id in {1, 2, 3}, self.nodes)),
            list(filter(lambda x: x.id in {5, 6, 9}, self.nodes))
        ]
        self.partitions2 = [
            list(filter(lambda x: x.id in {4, 5, 6}, self.nodes)),
            list(filter(lambda x: x.id in {1, 2, 3}, self.nodes)),
            list(filter(lambda x: x.id in {7, 8, 9}, self.nodes))
        ]

    def calculateValues(self):
        print("Partition Variant 1")
        print("GIL:", self.getGeneralizedInformationLossForGraph(self.nodes, self.partitions1))
        print("NGIL:", self.getNGIL(self.nodes, self.partitions1))
        print("\n---\n")

        print("Partition Variant 2")
        print("GIL:", self.getGeneralizedInformationLossForGraph(self.nodes, self.partitions2))
        print("NGIL:", self.getNGIL(self.nodes, self.partitions2))

    def searchCommonAncestor(self, searchItem, date_values):
        data = date_values.copy()
        return_value = None

        if searchItem in data.keys():
            return searchItem

        for key, value in data.items():
            if isinstance(value, dict):
                if searchItem in data[key].keys():
                    return_value = key
                else:
                    value = self.searchCommonAncestor(searchItem, data[key])
                    if value is None:
                        continue
                    else:
                        return_value = value

            elif isinstance(value, list):
                if str(searchItem) in value:
                    return_value = key
                else:
                    continue
            else:
                if value == searchItem or key == searchItem:
                    return_value = key
                else:
                    continue

        return return_value

    def gen_dict_extract(self, key, var):
        if hasattr(var, 'items'):
            for k, v in var.items():
                if v == key:
                    return k
                if isinstance(v, dict):
                    if self.gen_dict_extract(key, v) is None:
                        return None
                    for result in self.gen_dict_extract(key, v):
                        return result
                elif isinstance(v, list):
                    for d in v:
                        if self.gen_dict_extract(key, d) is None:
                            return None
                        for result in self.gen_dict_extract(key, d):
                            return result

    def depth(self, x):
        if type(x) is dict and x:
            return 1 + max(self.depth(x[a]) for a in x)
        if type(x) is list and x:
            return 1 + max(self.depth(a) for a in x)
        return 0



    # Definition 6
    def getNGIL(self, graph: [Node], partitions: [Node]):
        return self.getGeneralizedInformationLossForGraph(graph, partitions) / (
                len(self.nodes) * (len(self.categorical_identifiers) + len(self.numerical_identifiers)))

    # Definition 5
    def getGeneralizedInformationLossForGraph(self, graph: [Node], partitions: [[Node]]):
        sum = 0
        for partition in partitions:
            sum += self.getGeneralizationInformationLossForCluster(partition, graph)

        return sum

    # Definition 4
    def getGeneralizationInformationLossForCluster(self, cluster: [Node], graph: [Node]) -> float:
        clusterSize = len(cluster)
        numerical_sum = 0
        categorical_sum = 0

        for numerical_attribute in self.numerical_identifiers:
            cluster_interval = list(map(lambda nodeItem: nodeItem.value[numerical_attribute], cluster))
            size_cluster = max(cluster_interval) - min(cluster_interval)

            total_interval = list(map(lambda nodeItem: nodeItem.value[numerical_attribute], graph))
            size_total = max(total_interval) - min(total_interval)
            numerical_sum += size_cluster / size_total

        for categorical_attribute in self.categorical_identifiers:
            should_subtract = True
            with open(f'{categorical_attribute}_generalization_tree.json') as json_file:
                data = json.load(json_file)
                total_height = self.depth(data) - 1

                generalized_keys = []
                values = list(set(map(lambda x: x.value[categorical_attribute], cluster)))
                if len(values) == 1:
                    generalized_sum_key = values[0]
                    should_subtract = False
                else:
                    for node in cluster:
                        generalized_key = self.searchCommonAncestor(str(node.value[categorical_attribute]), data)
                        if generalized_key is not None:
                            generalized_keys.append(generalized_key)

                    generalized_sum_key = self.generalizeSum(data, generalized_keys)

                sub_generalization_tree = self.getSubtree(data, generalized_sum_key)
                sub_height = self.depth(sub_generalization_tree)
                if should_subtract is True:
                    sub_height -= 1

                categorical_sum += sub_height / total_height

        return clusterSize * (numerical_sum + categorical_sum)

    # Get subtree A(gen(cl)[Cj]
    def getSubtree(self, data, searchKey):
        if searchKey not in data.keys():
            for key, value in data.items():
                if isinstance(value, dict):
                    return self.getSubtree(data[key], searchKey)
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

    # Get common generalization for all values in list
    def generalizeSum(self, data, values: [str]):
        if len(set(values)) == 1:
            return list(set(values))[0]
        ancestors = []
        for item in values:
            ancestor = self.searchCommonAncestor(item, data)
            if ancestor is not None:
                ancestors.append(ancestor)

        if len(set(ancestors)) == 1:
            return list(set(ancestors))[0]
        else:
            for ancestor in ancestors.copy():
                ancestors.remove(ancestor)
                ancestors.append(self.searchCommonAncestor(ancestor, data))
                if len(set(ancestors)) == 1:
                    return list(set(ancestors))[0]
            return self.generalizeSum(data, ancestors)
