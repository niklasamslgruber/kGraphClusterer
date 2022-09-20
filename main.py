import pandas as pd
from dataHandler.graphGenerator import GraphGenerator
from engines.anonymizationEngine import AnonymizationEngine
from dataHandler.dataProcessor import DataProcessor
from dataHandler.datasets import Datasets
from engines.anonymizationType import AnonymizationType
from engines.gilEngine import GILEngine
from engines.graphEvaluationEngine import GraphEvaluationEngine
from engines.resultCollector import ResultCollector
from models.graph import Graph
import copy
import time
from engines.visualizationEngine import VisualizationEngine


def run(dataset: Datasets, alpha: float, beta: float, k: int, type: AnonymizationType, threshold: int,
        shouldPrint: bool = False):
    start_time = time.time()
    features: pd.DataFrame = DataProcessor.loadFeatures(dataset)
    features = features.sample(threshold, random_state=4)

    edges: pd.DataFrame = DataProcessor.loadEdges(dataset, threshold)
    assert len(edges["node1"].unique()) == len(features)

    graph = Graph.create(edges, features, dataset)

    partition = AnonymizationEngine(copy.copy(graph), alpha, beta, k, dataset, type).anonymize()
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
                                    len(partition.clusters), exec_time, type.value)

    ResultCollector(dataset).saveResult(result)


def runMultiple():
    for dataset in Datasets:
        if dataset != Datasets.ADULTS:
            continue
        for k in [2, 4, 6, 8, 10]:
            for alpha in [0, 0.5, 1]:
                for beta in [0, 0.5, 1]:
                    if alpha == 0 and beta == 0:
                        continue
                    for limit in [100, 300, 500]:
                        run(dataset, alpha, beta, k, AnonymizationType.DISCERNIBILITY_ALL, limit)


if __name__ == '__main__':
    print("Starting Clusterer...\n")

    # runMultiple()
    # run(Datasets.ADULTS, 1, 1, 3, AnonymizationType.SaNGreeA, 100)

    generator = GraphGenerator(dataset=Datasets.ADULTS, threshold=100)
    generator.generateEdges()
    generator.drawGraph()

    # run(Datasets.SAMPLE, 1, 0, 3, 10, AnonymizationType.SaNGreeA)
    #
    # visualizer = VisualizationEngine(Datasets.ADULTS, AnonymizationType.DISCERNIBILITY)
    # visualizer.drawGraph(100)
    # visualizer.plotNGIL()
    # visualizer.plotNSIL()
    # visualizer.plotPerformance()
