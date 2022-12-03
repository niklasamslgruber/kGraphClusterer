from dataHandler.datasets import Datasets


class Node:
    id: str
    degree: int
    relations: [int]
    value: dict
    cluster_id: str

    def __str__(self):
        return f'Id: {self.id}\n' \
               f'Degree: {self.degree}\n' \
               f'Relations: {list(map(lambda x: str(x), self.relations))}\n' \
               f'Value: {self.value}'

    def isEqual(self, dataset: Datasets, node):
        for identifier in dataset.getNumericalIdentifiers() + dataset.getCategoricalIdentifiers():
            if self.value[identifier] != node.value[identifier]:
                return False

        return True
