from config import FLAGS
from dataHandler.datasets import Datasets
from engines.anonymizationType import AnonymizationType
from engines.visualizationEngine import VisualizationEngine
from runner import Runner

if __name__ == '__main__':
    print("Starting Clusterer...\n")
    print("Params:", ', '.join(f'{k}={v}' for k, v in vars(FLAGS).items()))
    Runner().start()

    # VisualizationEngine(Datasets.ADULTS, 100).plotPerformance()
    # VisualizationEngine(Datasets.ADULTS, 100).plotNGIL()
    # VisualizationEngine(Datasets.ADULTS, 100).plotNSIL()


