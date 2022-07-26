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
    categorical_identifiers: [str] = ["zip"]

    def __init__(self):
        self.prepareData()
        self.sortByDegree()

        for categorical_attribute in self.categorical_identifiers:
            with open(f'{categorical_attribute}_generalization_tree.json') as json_file:
                data = json.load(json_file)
                # print(self.depth(data))
                # print(self.gen_dict_extract("482**", data))
                # print(self.searchCommonAncestor("482**", data))

                # print(list(data.keys()))
                # print(data)

    def searchCommonAncestor(self, searchItem, data):
        if searchItem in data.keys():
            return searchItem
        for key, value in data.items():
            if isinstance(value, dict):
                if searchItem in data[key].keys():
                    print("yo")
                    return key
                else:
                    # print(data[key], searchItem)
                    return self.searchCommonAncestor(searchItem, data[key])
            elif isinstance(value, list):
                if searchItem in value:
                    print("here")
                    return key
                else:
                    print(value, searchItem)
                    continue
            else:
                if value == searchItem or key == searchItem:
                    print("there")
                    return key
                else:
                    print("567")
                    continue
        print("here1")
        return None


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


    def prepareData(self):
        edges = pd.read_csv("edges.csv", names=["node1", "node2"], header=None)
        features = pd.read_csv("features.csv", index_col="node")
        for index, row in features.iterrows():
            node = Node()
            node.id = index
            node.relations = edges[edges["node1"] == index]["node2"].to_list()
            node.degree = len(node.relations)
            node.value = features.loc[index]
            self.nodes.append(node)

        assert len(self.nodes) == features.index.size
        partitions = [
            list(filter(lambda x: x.id in {1, 2, 3}, self.nodes)),
            list(filter(lambda x: x.id in {5, 6, 9}, self.nodes)),
            list(filter(lambda x: x.id in {4, 7, 8}, self.nodes))
        ]
        print(partitions)
        self.getGeneralizedInformationLossForGraph(self.nodes, partitions)


    def sortByDegree(self):
        s = []
        i = 1
        alpha = 1
        beta = 1
        clusters = []
        # n = set(self.edges_ordered_by_degree["node"].to_list())
        # print(n)

        # for index, row in self.edges_ordered_by_degree.iterrows():
        #     x_seed = row
        # x_star = min(alpha * self.getDistance())
        # print(index, row["node"], row["degree"])

    # def getNodeWithMinimalLoss(self, n, alpha, beta, graph, partition, cluster):
    #     minimum = (0, 0)
    #     for node in n:
    #         value = alpha * self.getNGIL(graph, partition) + beta * self.getClusterDistance(node, cluster)
    #         if value < minimum[0]:
    #             minimum = (value, node)
    #
    #     return minimum

    def getNGIL(self, graph: [Node], partitions: [Node]):
        return self.getGeneralizedInformationLossForGraph(graph, partitions) / (len(self.categorical_identifiers) + len(self.numerical_identifiers))

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
            with open(f'{categorical_attribute}_generalization_tree.json') as json_file:
                data = json.load(json_file)
                total_height = self.depth(data)

                generalized_keys = []
                for node in cluster:
                    print(node.value[categorical_attribute], data)
                    generalized_key = self.searchCommonAncestor(node.value[categorical_attribute], data)
                    generalized_keys.append(generalized_key)

                generalized_sum_key = self.generalizeSum(data, generalized_keys)
                sub_generalization_tree = self.getSubtree(data, generalized_sum_key)
                sub_height = self.depth(sub_generalization_tree)

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
            return data[searchKey]

    # Get common generalization for all values in list
    def generalizeSum(self, data, values: [str]):
        ancestors = []
        for item in values:
            ancestor = self.searchCommonAncestor(item, data)
            ancestors.append(ancestor)

        if len(set(ancestors)) == 1:
            return set(ancestors)
        else:
            for ancestor in ancestors.copy():
                ancestors.remove(ancestor)
                ancestors.append(self.searchCommonAncestor(ancestor, data))
                if len(set(ancestors)) == 1:
                    return set(ancestors)

            return self.generalizeSum(data, ancestors)


    # def getClusterDistance(self, node, cluster):
    #     return 4
    #
    # def getNodeDistance(self, x, y):
    #     return 2
