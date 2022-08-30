import pandas as pd
from dataProcessor import DataProcessor
import random
from os.path import exists


class GraphGenerator:

    @staticmethod
    def generateRandomEdges(dataset: DataProcessor.Dataset, num: int, limit: int = -1, force: bool = False):
        edge_file = dataset.getEdgePath()

        if exists(edge_file) and not force:
            print("Edge File already exists and will not be generated again. Use the `force` paramter to re-generate edges again.")
            return

        frame = DataProcessor.loadData(dataset)

        if limit != -1 and limit < len(frame):
            frame = frame.head(limit)

        entries = len(frame)

        edges: [[int]] = []

        for count in range(0, num):
            node1 = random.randint(0, entries - 1)
            node2 = node1
            while node1 == node2:
                node2 = random.randint(0, entries - 1)

            assert node1 != node2

            edge1 = [node1, node2]
            edge2 = [node2, node1]

            edges.append(edge1)
            edges.append(edge2)

        assert len(edges) == 2 * num

        dataset.createDirectoryIfNotExists()

        frame = pd.DataFrame(edges)
        frame = frame.drop_duplicates()

        print(f"Generated {len(frame)} edges for dataset {dataset.name}.\n"
              f"Note:\n"
              f"\t(1) Each edge is added twice to be symmetric (total maximum number of edges: 2 * num)\n"
              f"\t(2) The actual number of edges could be lower than your provided threshold due to the automatic removal of duplicates.")

        frame.to_csv(edge_file, index=False, header=False)
