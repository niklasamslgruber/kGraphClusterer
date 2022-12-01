import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from constants import RANDOM_SEED
from dataHandler.dataProcessor import DataProcessor
from dataHandler.datasets import Datasets
from dataHandler.graphGenerator import GraphGenerator
from engines.anonymizationType import AnonymizationType
from models.partition import Partition


class VisualizationEngine:
    dataset: Datasets
    threshold: int

    FONTSIZE = 30
    LINE_WIDTH = 7

    def __init__(self, dataset: Datasets, threshold: int):
        self.dataset = dataset
        self.threshold = threshold

    def plotNGIL(self):
        fig, ax = self.__plotMetrics("ngil", "NGIL")
        fig.savefig(f'{self.dataset.getImagePath()}/ngil.png', dpi=200)

        for size in [100, 300, 500, 1000]:
            fig, ax = self.__plotMetricsSized(size, "ngil", "NGIL", excluded=[AnonymizationType.MODULARITY, AnonymizationType.SILHOUETTE, AnonymizationType.GRAPH_PERFORMANCE, AnonymizationType.PURITY])
            fig.savefig(f'{self.dataset.getImagePath()}/ngil{size}_traditional.png', dpi=100)

            fig, ax = self.__plotMetricsSized(size, "ngil", "NGIL",
                                              excluded=[AnonymizationType.DISCERNIBILITY, AnonymizationType.PRECISION,
                                                        AnonymizationType.CLASSIFICATION_METRIC, AnonymizationType.NORMALIZED_CERTAINTY_PENALTY, AnonymizationType.ENTROPY])
            fig.savefig(f'{self.dataset.getImagePath()}/ngil{size}_graph_clusters.png', dpi=100)

    def plotNSIL(self):
        fig, ax = self.__plotMetrics("nsil", "NSIL")
        fig.savefig(f'{self.dataset.getImagePath()}/nsil.png', dpi=200)

        for size in [100, 300, 500, 1000]:
            fig, ax = self.__plotMetricsSized(size, "nsil", "NSIL", excluded=[AnonymizationType.MODULARITY, AnonymizationType.SILHOUETTE, AnonymizationType.GRAPH_PERFORMANCE, AnonymizationType.PURITY])
            fig.savefig(f'{self.dataset.getImagePath()}/nsil{size}_traditional.png', dpi=200)

            fig, ax = self.__plotMetricsSized(size, "nsil", "NSIL",
                                              excluded=[AnonymizationType.DISCERNIBILITY, AnonymizationType.PRECISION,
                                                        AnonymizationType.CLASSIFICATION_METRIC,
                                                        AnonymizationType.NORMALIZED_CERTAINTY_PENALTY,
                                                        AnonymizationType.ENTROPY])
            fig.savefig(f'{self.dataset.getImagePath()}/nsil{size}_graph_clusters.png', dpi=100)

    def plotCGIL(self):
        fig, ax = self.__plotMetrics("cgil", "CGIL")
        fig.savefig(f'{self.dataset.getImagePath()}/cgil.png', dpi=200)

        for size in [100, 300, 500, 1000]:
            fig, ax = self.__plotMetricsSized(size, "cgil", "CGIL", excluded=[])
            fig.savefig(f'{self.dataset.getImagePath()}/cgil{size}.png', dpi=200)

    def plotPerformance(self):
        fig, ax = self.__plotPerformance("time", "Time [s]", excluded=[AnonymizationType.MODULARITY, AnonymizationType.SILHOUETTE, AnonymizationType.GRAPH_PERFORMANCE, AnonymizationType.PURITY])
        fig.savefig(f'{self.dataset.getImagePath()}/time_traditional.png', dpi=200)

        fig, ax = self.__plotPerformance("time", "Time [s]",
                                         excluded=[AnonymizationType.DISCERNIBILITY, AnonymizationType.PRECISION,
                                                        AnonymizationType.CLASSIFICATION_METRIC,
                                                        AnonymizationType.NORMALIZED_CERTAINTY_PENALTY,
                                                        AnonymizationType.ENTROPY])
        fig.savefig(f'{self.dataset.getImagePath()}/time_graph_clusters.png', dpi=200)

    def __plotMetricsSized(self, size: int, y: str, y_label: str, excluded: [AnonymizationType] = None):
        frame = pd.read_csv(self.dataset.getResultsPath(), header=0)

        if excluded is not None:
            for type in excluded:
                frame = frame[frame["method"] != type.value]

        fig, ax = plt.subplots(2, 2, figsize=(50, 50), sharey=True)

        a_b_pairs = [(1, 0), (1, 0.5), (0.5, 1), (1, 1)]

        for key, grp in frame.groupby(["method"]):
            col = 0
            row = 0

            if key == AnonymizationType.NORMALIZED_CERTAINTY_PENALTY.value:
                key = "NPC"

            for index, (alpha, beta) in enumerate(a_b_pairs):
                small = grp[(grp["size"] == size) & (grp["alpha"] == alpha) & (grp["beta"] == beta)].sort_values(by=['k'])
                ax[col, row] = small.plot(ax=ax[col, row], kind='line', x='k', y=y, label=key, linewidth=self.LINE_WIDTH, legend=0 if index != 0 else 1)
                ax[col, row].set_title(f"alpha = {alpha}, beta = {beta}", fontsize=self.FONTSIZE)

                if index == 1:
                    col += 1
                    row = 0
                else:
                    row += 1

        for axItem in ax.ravel():
            axItem.legend(loc=2, prop={'size': self.FONTSIZE})
            axItem.set(xlabel='k', ylabel=y_label)
            axItem.spines["left"].set_linewidth(self.LINE_WIDTH)
            axItem.spines["bottom"].set_linewidth(self.LINE_WIDTH)
            axItem.spines["right"].set_linewidth(self.LINE_WIDTH)
            axItem.spines["top"].set_linewidth(self.LINE_WIDTH)
            axItem.tick_params(axis='both', labelsize=self.FONTSIZE)
            axItem.set_xlabel("k", fontsize=self.FONTSIZE)
            axItem.set_ylabel(y_label, fontsize=self.FONTSIZE)

            if y == "ngil":
                axItem.set_ylim(0, 1)
                axItem.set_yticks(np.arange(0, 1, 0.05))

        plt.tight_layout()

        return fig, ax

    def __plotMetrics(self, y: str, y_label: str):
        frame = pd.read_csv(self.dataset.getResultsPath(), header=0)

        fig, ax = plt.subplots(5, 3, figsize=(40, 40))

        for key, grp in frame.groupby(["method"]):
            index = 0
            a_b_pairs = [(1, 0), (1, 0.5), (0.5, 1), (0.5, 0.5), (1, 1)]
            for (alpha, beta) in a_b_pairs:
                small = grp[(grp["size"] == 100) & (grp["alpha"] == alpha) & (grp["beta"] == beta)].sort_values(by=['k'])
                medium = grp[(grp["size"] == 300) & (grp["alpha"] == alpha) & (grp["beta"] == beta)].sort_values(by=['k'])
                big = grp[(grp["size"] == 500) & (grp["alpha"] == alpha) & (grp["beta"] == beta)].sort_values(by=['k'])

                if key == AnonymizationType.NORMALIZED_CERTAINTY_PENALTY.value:
                    key = "NPC"

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
                axItem.set_yticks(np.arange(0, 1, 0.05))

            if y == "nsil":
                axItem.set_ylim(0, 0.15)
                axItem.set_yticks(np.arange(0, 0.15, 0.05))

        plt.tight_layout()

        return fig, ax

    def __plotPerformance(self, y: str, y_label: str, excluded: [AnonymizationType] = None):
        frame = pd.read_csv(self.dataset.getResultsPath(), header=0)

        fig, ax = plt.subplots(2, 3, figsize=(20, 40), sharey=True)

        if excluded is not None:
            for type in excluded:
                frame = frame[frame["method"] != type.value]

        chosen_k = 4
        chosen_k2 = 6
        chosen_k3 = 10

        for key, grp in frame.groupby(["method"]):
            index = 0
            a_b_pairs = [(1, 0.5), (1, 1)]
            for (alpha, beta) in a_b_pairs:
                small = grp[(grp["k"] == chosen_k) & (grp["alpha"] == alpha) & (grp["beta"] == beta)].sort_values(by=['k'])
                medium = grp[(grp["k"] == chosen_k2) & (grp["alpha"] == alpha) & (grp["beta"] == beta)].sort_values(by=['k'])
                big = grp[(grp["k"] == chosen_k3) & (grp["alpha"] == alpha) & (grp["beta"] == beta)].sort_values(by=['k'])

                if key == AnonymizationType.NORMALIZED_CERTAINTY_PENALTY.value:
                    key = "NPC"

                if len(small) > 0:
                    ax[index, 0] = small.plot(ax=ax[index, 0], kind='line', x='size', y="time", label=key, linewidth=self.LINE_WIDTH)
                    ax[index, 0].set_title(f"k = {chosen_k}, alpha = {alpha}, beta = {beta}", fontsize=self.FONTSIZE)

                if len(medium) > 0:
                    ax[index, 1] = medium.plot(ax=ax[index, 1], kind='line', x='size', y="time", label=key, linewidth=self.LINE_WIDTH)
                    ax[index, 1].set_title(f"k = {chosen_k2}, alpha = {alpha}, beta = {beta}", fontsize=self.FONTSIZE)

                if len(big) > 0:
                    ax[index, 2] = big.plot(ax=ax[index, 2], kind='line', x='size', y="time", label=key, linewidth=self.LINE_WIDTH)
                    ax[index, 2].set_title(f"k = {chosen_k3}, alpha = {alpha}, beta = {beta}", fontsize=self.FONTSIZE)

                index += 1

        for axItem in ax.ravel():
            axItem.legend(loc=2, prop={'size': self.FONTSIZE})
            axItem.set(xlabel='Dataset Size', ylabel=y_label)
            axItem.xaxis.set_ticks([100, 300, 500, 1000])
            axItem.spines["left"].set_linewidth(self.LINE_WIDTH)
            axItem.spines["bottom"].set_linewidth(self.LINE_WIDTH)
            axItem.spines["right"].set_linewidth(self.LINE_WIDTH)
            axItem.spines["top"].set_linewidth(self.LINE_WIDTH)
            axItem.tick_params(axis='both', labelsize=self.FONTSIZE)
            axItem.set_xlabel("k", fontsize=self.FONTSIZE)
            axItem.set_ylabel(y_label, fontsize=self.FONTSIZE)

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

