import pandas as pd
from dataHandler.graphGenerator import GraphGenerator
from engines.anonymizationEngine import AnonymizationEngine
from dataHandler.dataProcessor import DataProcessor
from dataHandler.datasets import Datasets
from engines.gilEngine import GILEngine
from engines.graphEvaluationEngine import GraphEvaluationEngine
from engines.resultCollector import ResultCollector
from models.graph import Graph
import copy
import time
from engines.visualizationEngine import VisualizationEngine


def run(dataset: Datasets, alpha: float, beta: float, k: int, vertex_degree: int, limit: int = -1,
          shouldPrint: bool = False):
    start_time = time.time()
    features: pd.DataFrame = DataProcessor.loadFeatures(dataset)

    if limit != -1:
        features = features.sample(limit, random_state=1)

    frame = GraphGenerator.generateRandomEdges(dataset, features, vertex_degree=vertex_degree, force=True)
    edges: pd.DataFrame = frame

    graph = Graph.create(edges, features, dataset)

    partition = AnonymizationEngine(copy.copy(graph), alpha, beta, k, dataset).anonymize()
    nsil = GraphEvaluationEngine(partition, graph).getNSIL()
    ngil = GILEngine(graph, partition, dataset).getNGIL()
    end_time = time.time()
    exec_time = round(end_time - start_time, 4)

    if shouldPrint:
        print("\n-----")
        print(f"Generated Clusters for dataset {dataset.name} (k = {k}, alpha = {alpha}, beta = {beta}):")
        for (index, cluster) in enumerate(partition.clusters):
            print(f"\tCluster {index + 1}:", cluster.getIds())

        print("\nStatistics:")
        print("\tNSIL:", nsil)
        print("\tNGIL:", ngil)
        print("Execution time", exec_time, "s")

    result = ResultCollector.Result(k, alpha, beta, len(features), len(edges), round(ngil, 4), round(nsil, 4),
                                    len(partition.clusters), exec_time, "SaNGreeA", vertex_degree)

    ResultCollector(dataset).saveResult(result)


def runMultiple():
    for dataset in Datasets:
        for k in [2, 3, 4, 5, 6, 10]:
            for alpha in [0, 0.5, 1]:
                for beta in [0, 0.5, 1]:
                    if alpha == 0 and beta == 0:
                        continue

                    for limit in [100, 300, 500]:
                        for degree in [3, 5, 10, 20]:
                            run(dataset, alpha, beta, k, degree, limit=limit)


if __name__ == '__main__':
    print("Starting Clusterer...\n")

    visualizer = VisualizationEngine(Datasets.ADULTS)
    # visualizer.plotNGIL()
    # visualizer.plotNSIL()
    visualizer.plotPerformance()
