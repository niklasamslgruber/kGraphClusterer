import pandas as pd
from dataHandler.dataProcessor import DataProcessor
import random
from os.path import exists

from dataHandler.datasets import Datasets


class GraphGenerator:

    @staticmethod
    def generateRandomEdges(dataset: Datasets, features: pd.DataFrame, vertex_degree: int, force: bool = False):
        edge_file = dataset.getEdgePath()

        if exists(edge_file) and not force:
            print("Edge File already exists and will not be generated again. Use the `force` paramter to re-generate edges again.")
            return

        edges: [[int]] = []

        for count in features.index:
            node1 = count

            for degree in range(0, vertex_degree):
                if degree == count:
                    continue
                node2 = random.choice(features.index)

                while node1 == node2:
                    node2 = random.choice(features.index)

                assert node1 != node2

                edge1 = [node1, node2]
                edge2 = [node2, node1]

                edges.append(edge1)
                edges.append(edge2)

        # assert len(edges) == 2 * entries * vertex_degree

        dataset.createDirectoryIfNotExists()

        frame = pd.DataFrame(edges)
        # frame = frame.drop_duplicates()

        # print(f"Generated {len(frame)} edges for dataset {dataset.name}.\n")
        # Note:
        #   (1) Each edge is added twice to be symmetric (total maximum number of edges: 2 * num)
        #   (2) The actual number of edges could be lower than your provided threshold due to the automatic removal of duplicates.

        # frame.to_csv(edge_file, index=False, header=False)
        frame.columns = ["node1", "node2"]
        return frame
