import matplotlib.pyplot as plt
import pandas as pd
import random
from os.path import exists
import networkx as nx

from dataHandler.datasets import Datasets


class GraphGenerator:
    dataset: Datasets
    threshold: int
    path: str

    def __init__(self, dataset: Datasets, threshold: int = 100):
        self.dataset = dataset
        self.threshold = threshold
        self.path = f"{dataset.getDirectory()}/edges_{threshold}.csv"

    @staticmethod
    def generateRandomEdges(dataset: Datasets, features: pd.DataFrame, vertex_degree: int, force: bool = False):
        edge_file = dataset.getEdgePath()

        if exists(edge_file) and not force:
            print(
                "Edge File already exists and will not be generated again. Use the `force` parameter to re-generate edges again.")
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

    def loadGraphDataset(self):
        frame = pd.read_csv(f"{Datasets.BANK_CLIENTS.getDirectory()}/2019-10_03.csv", low_memory=False).dropna()
        for column in frame.keys():
            if column == "source_address" or column == "destination_address":
                continue
            frame = frame.drop(column, axis=1)

        source_series = frame["source_address"].value_counts(ascending=False)
        count_source_frame = pd.DataFrame(source_series).reset_index()
        count_source_frame.columns = ["id", "count"]
        count_source_frame = count_source_frame[count_source_frame["count"] >= 5]

        destination_series = frame["destination_address"].value_counts(ascending=False)
        count_destination_frame = pd.DataFrame(destination_series).reset_index()
        count_destination_frame.columns = ["id", "count"]
        count_destination_frame = count_destination_frame[count_destination_frame["count"] >= 5]

        count_frame = pd.concat([count_destination_frame, count_source_frame])

        ids = count_frame["id"].unique()
        random.Random(4).shuffle(ids)

        edge_frame = pd.DataFrame()
        added_ids = []
        for id in ids:
            if len(added_ids) >= self.threshold:
                break

            print(len(added_ids))
            if len(added_ids) > self.threshold * 0.8:
                filtered_source_frame = frame[
                    (frame["source_address"] == id) & (frame["destination_address"].isin(added_ids)) & (
                            frame["destination_address"] != id)]
            else:
                filtered_source_frame = frame[(frame["source_address"] == id) & (frame["destination_address"] != id)]

            if filtered_source_frame.empty is False and len(added_ids) < self.threshold:
                edge_frame = pd.concat([edge_frame, filtered_source_frame])
                if id not in added_ids:
                    added_ids.append(id)
                for (index, row) in filtered_source_frame.iterrows():
                    if row["destination_address"] not in added_ids:
                        added_ids.append(row["destination_address"])

            # Other direction

            if len(added_ids) > self.threshold * 0.8:
                filtered_destination_frame = frame[
                    (frame["destination_address"] == id) & (frame["source_address"].isin(added_ids)) & (
                            frame["source_address"] != id)]
            else:
                filtered_destination_frame = frame[
                    (frame["destination_address"] == id) & (frame["source_address"] != id)]

            if filtered_destination_frame.empty is False and len(added_ids) < self.threshold:
                edge_frame = pd.concat([edge_frame, filtered_destination_frame])
                if id not in added_ids:
                    added_ids.append(id)
                for (index, row) in filtered_destination_frame.iterrows():
                    if row["source_address"] not in added_ids:
                        added_ids.append(row["source_address"])

        for (index, row) in edge_frame.iterrows():
            if row["source_address"] not in added_ids:
                assert False, f"Source includes ids ({row['destination_address']}) that are unknown"
            if row["destination_address"] not in added_ids:
                assert False, f"Destination includes ids ({row['destination_address']}) that are unknown"

        edge_frame = edge_frame.drop_duplicates()

        for (index, row) in edge_frame.iterrows():
            if edge_frame[(edge_frame["destination_address"] == row["source_address"]) &
                          (edge_frame["source_address"] == row["destination_address"])].empty is False:
                edge_frame.drop(index, axis=0, inplace=True)

        print(edge_frame)
        print(edge_frame["source_address"].value_counts())
        print(edge_frame["destination_address"].value_counts())

        edge_frame.to_csv(self.path, index=False)

    def drawGraph(self):
        edge_frame = pd.read_csv(self.path)

        edge_frame_copy = edge_frame.copy()
        edge_frame_copy.columns = ["destination_address", "source_address"]

        G = nx.Graph()
        for (index, row) in edge_frame.iterrows():
            G.add_node(row["source_address"])
            G.add_node(row["destination_address"])
            G.add_edge(row["destination_address"], row["source_address"])

        print(len(edge_frame.drop_duplicates()))

        print(len(G.nodes), len(G.edges))
        assert len(G.nodes) >= self.threshold

        pos = nx.spring_layout(G, k=0.1, seed=4)
        nx.draw(G, pos=pos, with_labels=False, node_size=20, width=0.5)
        plt.show()

    # def associateFeatures(self, ids: [str], threshold: int = 1000):
