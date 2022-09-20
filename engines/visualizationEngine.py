import matplotlib.pyplot as plt
import pandas as pd
from dataHandler.datasets import Datasets
from engines.anonymizationType import AnonymizationType


class VisualizationEngine:
    dataset: Datasets
    type: AnonymizationType

    def __init__(self, dataset: Datasets, type: AnonymizationType):
        self.dataset = dataset
        self.type = type

    def plotNGIL(self):
        fig, ax = self.__plotMetrics("ngil", "NGIL")
        fig.savefig(f'{self.dataset.getImagePath()}/ngil_{self.type.name}.png', dpi=200)

    def plotNSIL(self):
        fig, ax = self.__plotMetrics("nsil", "NSIL")
        fig.savefig(f'{self.dataset.getImagePath()}/nsil_{self.type.name}.png', dpi=200)

    def plotPerformance(self):
        fig, ax = self.__plotMetrics("time", "Time [s]")
        fig.savefig(f'{self.dataset.getImagePath()}/time_{self.type.name}.png', dpi=200)

    def __plotMetrics(self, y: str, y_label: str):
        frame = pd.read_csv(self.dataset.getResultsPath(), header=0)

        fig, ax = plt.subplots(8, 3, figsize=(20, 40))

        frame = frame[frame["method"] == self.type.value]

        for key, grp in frame.groupby(["vertex_degree"]):
            index = 0
            for alpha in [0, 0.5, 1]:
                for beta in [0, 0.5, 1]:
                    if alpha == beta and beta == 0:
                        continue

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
            if y != "time":
                axItem.set_ylim(0, 1)

        plt.tight_layout()

        return fig, ax
