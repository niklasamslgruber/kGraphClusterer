class Node:
    id: int
    degree: int
    relations: [int]
    value: dict

    def __str__(self):
        return f'Id: {self.id}\n' \
               f'Degree: {self.degree}\n' \
               f'Relations: {list(map(lambda x: str(x), self.relations))}\n' \
               f'Value: {self.value}'
