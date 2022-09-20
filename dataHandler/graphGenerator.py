import pandas as pd
import random
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataHandler.dataProcessor import DataProcessor
from dataHandler.datasets import Datasets


class GraphGenerator:
    dataset: Datasets
    threshold: int
    path: str

    def __init__(self, dataset: Datasets, threshold: int = 100):
        self.dataset = dataset
        self.threshold = threshold

    def generateEdges(self):
        frame = pd.read_csv("data/bitcoin2019-10_03.csv", low_memory=False).dropna()
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
        with tqdm(total=self.threshold) as pbar:
            count = 0
            for id in ids:
                count += 1
                pbar.set_postfix(str=f"Processed IDs: {count}/{len(ids)}")
                if id in added_ids:
                    continue

                if len(added_ids) >= self.threshold:
                    break

                if len(added_ids) > self.threshold * 0.8:
                    filtered_source_frame = frame[
                        ((frame["source_address"] == id) & (frame["destination_address"].isin(added_ids)) & (
                                frame["destination_address"] != id))
                        |
                        ((frame["destination_address"] == id) & (frame["source_address"].isin(added_ids)) & (
                                frame["source_address"] != id))
                        ]
                else:
                    filtered_source_frame = frame[
                        ((frame["source_address"] == id) & (frame["destination_address"] != id)) | (
                                    (frame["destination_address"] == id) & (frame["source_address"] != id))]

                if filtered_source_frame.empty is False and len(added_ids) < self.threshold:
                    edge_frame = pd.concat([edge_frame, filtered_source_frame])
                    if id not in added_ids:
                        added_ids.append(id)
                        pbar.update()

                    for (index, row) in filtered_source_frame.iterrows():
                        if row["destination_address"] not in added_ids:
                            added_ids.append(row["destination_address"])
                            pbar.update()
                        if row["source_address"] not in added_ids:
                            added_ids.append(row["source_address"])
                            pbar.update()

        edge_frame = edge_frame.drop_duplicates()

        for (index, row) in edge_frame.iterrows():
            if row["source_address"] not in added_ids:
                assert False, f"Source includes ids ({row['destination_address']}) that are unknown"
            if row["destination_address"] not in added_ids:
                assert False, f"Destination includes ids ({row['destination_address']}) that are unknown"

            if edge_frame[(edge_frame["destination_address"] == row["source_address"]) &
                          (edge_frame["source_address"] == row["destination_address"])].empty is False:
                edge_frame.drop(index, axis=0, inplace=True)

        edge_frame_copy = edge_frame.copy()
        double_edge_frame = pd.concat([edge_frame, edge_frame_copy])
        assert len(double_edge_frame) == len(edge_frame) + len(edge_frame_copy)

        final_edge_frame = self.__generateMissingClusterConnections(double_edge_frame)
        final_edge_frame.to_csv(self.dataset.getEdgePath(self.threshold), index=False, header=False)

    # Make sure that the edge file does not contain unconnected subgraphs, but ONE connect graph
    def __generateMissingClusterConnections(self, edge_frame: pd.DataFrame):
        G = GraphGenerator.createGraphFromDataframe(edge_frame)

        assert len(G.nodes) >= self.threshold

        sub_graphs = sorted(list(nx.connected_components(G)))

        factor = 0.125
        new_edges = []
        for (index, cluster) in enumerate(sub_graphs):
            for sub_index in range(index, len(sub_graphs)):
                partner_cluster = sorted(list(sub_graphs[sub_index]))
                partner_cluster_factor = int(len(partner_cluster) * factor)
                partner_cluster_random_edges = random.Random(4).choices(partner_cluster, k=partner_cluster_factor)

                own_cluster = sorted(list(cluster))
                own_cluster_factor = int(len(own_cluster) * factor)
                own_cluster_random_edges = random.Random(4).choices(own_cluster, k=own_cluster_factor)

                for partner_node in partner_cluster_random_edges:
                    for own_node in own_cluster_random_edges:
                        if own_node != partner_node and partner_node not in own_cluster:
                            G.add_edge(own_node, partner_node)
                            new_edges.append([own_node, partner_node])
                            new_edges.append([partner_node, own_node])

        assert len(list(nx.connected_components(G))) == 1, "New graph still contains unconnected cluster"

        new_frame = pd.DataFrame(new_edges)
        updated_edge_frame = pd.concat([edge_frame, new_frame])

        return updated_edge_frame

    def drawGraph(self):
        edge_frame = DataProcessor.loadEdges(self.dataset, self.threshold)
        G = GraphGenerator.createGraphFromDataframe(edge_frame)
        pos = nx.spring_layout(G, k=0.1, seed=4)
        nx.draw(G, pos=pos, with_labels=False, node_size=20, width=0.5)
        plt.show()

    @staticmethod
    def createGraphFromDataframe(edge_frame: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        edge_frame.columns = ["node1", "node2"]
        for (index, row) in edge_frame.iterrows():
            G.add_node(row["node1"])
            G.add_node(row["node2"])
            G.add_edge(row["node1"], row["node2"])

        return G
    # def associateFeatures(self, ids: [str], threshold: int = 1000):
