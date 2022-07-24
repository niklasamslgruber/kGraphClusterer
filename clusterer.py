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
                print(self.search("48201", data))
                # print(list(data.keys()))
                # print(data)

    def search(self, searchItem, data):
        for key, value in data.items():
            if isinstance(value, dict):
                if searchItem in data[key].keys():
                    return key
                else:
                    return self.search(searchItem, data[key])
            elif isinstance(value, list):
                if searchItem in value:
                    return key
                else:
                    continue
            else:
                if value == searchItem or key == searchItem:
                    return key
                else:
                    continue


    def gen_dict_extract(self, key, var):
        if hasattr(var, 'items'):
            for k, v in var.items():
                print(k, v)
                if v == key:
                    return k
                if isinstance(v, dict):
                    print("here")
                    if self.gen_dict_extract(key, v) is None:
                        return None
                    print(self.gen_dict_extract(key, v))
                    for result in self.gen_dict_extract(key, v):
                        return result
                elif isinstance(v, list):
                    print("there")
                    for d in v:
                        print(d)
                        print(self.gen_dict_extract(key, d))
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

    def getNGIL(self, graph, partition):
        return 3

    # Definition 4
    def getGeneralizationInformationLoss(self, cluster: [Node], graph: [Node]) -> float:
        clusterSize = len(cluster)
        numerical_sum = 0
        categorical_sum = 0

        for numerical_attribute in self.numerical_identifiers:
            cluster_interval = list(map(lambda node: node.value[numerical_attribute], cluster))
            size_cluster = max(cluster_interval) - min(cluster_interval)

            total_interval = list(map(lambda node: node.value[numerical_attribute], graph))
            size_total = max(total_interval) - min(total_interval)

            numerical_sum += size_cluster / size_total

        for categorical_attribute in self.categorical_identifiers:
            with open(f'{categorical_attribute}_generalization_tree.json') as json_file:
                data = json.load(json_file)
                total_height = self.depth(data)

                generalized_keys = []
                for node in cluster:
                    generalized_key = self.search(node.value[categorical_attribute], data)
                    generalized_keys.append(generalized_key)

                generalized_sum_key = self.generalizeSum(generalized_keys)
                sub_generalization_tree = self.getSubtree(data, generalized_sum_key)
                sub_height = self.depth(sub_generalization_tree)

                categorical_sum += sub_height / total_height

        return clusterSize * (numerical_sum + categorical_sum)

    # Get subtree A(gen(cl)[Cj]
    def getSubtree(self, data, key):
        return data

    # Get common generalization for all values in list
    def generalizeSum(self, values: [str]):
        return "X"


    # def getClusterDistance(self, node, cluster):
    #     return 4
    #
    # def getNodeDistance(self, x, y):
    #     return 2
