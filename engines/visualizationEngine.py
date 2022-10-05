import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from constants import RANDOM_SEED
from dataHandler.dataProcessor import DataProcessor
from dataHandler.datasets import Datasets
from dataHandler.graphGenerator import GraphGenerator
from models.partition import Partition


class VisualizationEngine:
    dataset: Datasets
    threshold: int

    def __init__(self, dataset: Datasets, threshold: int):
        self.dataset = dataset
        self.threshold = threshold

    def plotNGIL(self):
        fig, ax = self.__plotMetrics("ngil", "NGIL")
        fig.savefig(f'{self.dataset.getImagePath()}/ngil.png', dpi=200)

    def plotNSIL(self):
        fig, ax = self.__plotMetrics("nsil", "NSIL")
        fig.savefig(f'{self.dataset.getImagePath()}/nsil.png', dpi=200)

    def plotPerformance(self):
        fig, ax = self.__plotMetrics("time", "Time [s]")
        fig.savefig(f'{self.dataset.getImagePath()}/time.png', dpi=200)

    def __plotMetrics(self, y: str, y_label: str):
        frame = pd.read_csv(self.dataset.getResultsPath(), header=0)

        fig, ax = plt.subplots(5, 3, figsize=(20, 40))

        for key, grp in frame.groupby(["method"]):
            index = 0
            a_b_pairs = [(1, 0), (1, 0.5), (0.5, 1), (0.5, 0.5), (1, 1)]
            for (alpha, beta) in a_b_pairs:
                small = grp[(grp["size"] == 100) & (grp["alpha"] == alpha) & (grp["beta"] == beta)]
                medium = grp[(grp["size"] == 300) & (grp["alpha"] == alpha) & (grp["beta"] == beta)]
                big = grp[(grp["size"] == 500) & (grp["alpha"] == alpha) & (grp["beta"] == beta)]

                if len(small) > 0:
                    ax[index, 0] = small.plot(ax=ax[index, 0], kind='line', x='k', y=y, label=key)
                    ax[index, 0].set_title(f"Graph Size = 100, alpha = {alpha}, beta = {beta}")

                if len(medium) > 0:
                    ax[index, 1] = medium.plot(ax=ax[index, 1], kind='line', x='k', y=y, label=key)
                    ax[index, 1].set_title(f"Graph Size = 300, alpha = {alpha}, beta = {beta}")

                if len(big) > 0:
                    ax[index, 2] = big.plot(ax=ax[index, 2], kind='line', x='k', y=y, label=key)
                    ax[index, 2].set_title(f"Graph Size = 500, alpha = {alpha}, beta = {beta}")

                index += 1

        for axItem in ax.ravel():
            axItem.legend(loc=2, prop={'size': 6})
            axItem.set(xlabel='k', ylabel=y_label)
            if y == "ngil":
                axItem.set_ylim(0, 1)

            if y == "nsil":
                axItem.set_ylim(0, 0.15)

        plt.tight_layout()

        return fig, ax

    def drawGraph(self, partition: Partition):
        G = nx.Graph()
        for cluster in partition.clusters:
            for node in cluster.nodes:
                G.add_node(node.cluster_id)

                for relation in node.relations:
                    receivingClusters = list(filter(lambda x: relation in x.getIds(), partition.clusters))

                    for receivingCluster in receivingClusters:
                        if receivingCluster.id == cluster.id:
                            continue
                        G.add_node(receivingCluster.id)
                        G.add_edge(node.cluster_id, receivingCluster.id)

        assert len(G.nodes) == len(partition.clusters), f"{G.nodes} {self.threshold}"

        pos = nx.spring_layout(G, k=0.1, seed=RANDOM_SEED)
        nx.draw(G, pos=pos, with_labels=False, node_size=20, width=0.5)
        plt.savefig(f"{self.dataset.getImagePath()}/anonymousGraph_{self.threshold}.png", dpi=200)

    def drawInitialGraph(self):
        edge_frame = DataProcessor.loadEdges(self.dataset, self.threshold)
        G = GraphGenerator.createGraphFromDataframe(edge_frame)
        pos = nx.spring_layout(G, k=0.1, seed=RANDOM_SEED)
        nx.draw(G, pos=pos, with_labels=False, node_size=20, width=0.5)
        plt.savefig(f"{self.dataset.getImagePath()}/initialGraph_{self.threshold}.png", dpi=200)

