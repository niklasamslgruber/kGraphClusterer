import pandas as pd
from config import FLAGS
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


class Runner:
    type: AnonymizationType
    threshold: int
    dataset: Datasets

    def __init__(self):
        self.type = AnonymizationType.getCase(FLAGS.method)
        self.threshold = FLAGS.size
        self.dataset = Datasets.getCase(FLAGS.dataset)

    def start(self):
        if FLAGS.generate_edges is not None:
            print(f"Generating {FLAGS.generate_edges} edges...")
            self.generateEdges()
        else:
            Runner.run(self.dataset, FLAGS.alpha, FLAGS.beta, FLAGS.k, self.type, FLAGS.size)

        if FLAGS.plot:
            print("Plotting results...")
            self.visualizeResults()

    @staticmethod
    def run(dataset: Datasets, alpha: float, beta: float, k: int, type: AnonymizationType, threshold: int, shouldPrint: bool = False):
        start_time = time.time()
        features: pd.DataFrame = DataProcessor.loadFeatures(dataset, threshold)

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
            print(f"Generated Clusters for dataset {dataset.name}:")
            for (index, cluster) in enumerate(partition.clusters):
                print(f"\tCluster {index + 1}:", cluster.getIds())

            print("\nStatistics:")
            print("\tNSIL:", nsil)
            print("\tNGIL:", ngil)
            print("Execution time", exec_time, "s")

        result = ResultCollector.Result(k, alpha, beta, len(features), len(edges), round(ngil, 4), round(nsil, 4),
                                        len(partition.clusters), exec_time, type.value)

        ResultCollector(dataset).saveResult(result)

    def runMultiple(self):
        a_b_pairs = [(0, 1), (1, 0), (1, 0.5), (0.5, 1), (0.5, 0.5), (1, 1)]
        for k in [2, 4, 6, 8, 10]:
            for (alpha, beta) in a_b_pairs:
                for limit in [100, 300, 500]:
                    self.run(self.dataset, alpha, beta, k, self.type, limit)

    def runMetrics(self):
        for metric in AnonymizationType:
            self.run(self.dataset, FLAGS.alpha, FLAGS.beta, FLAGS.k, metric, FLAGS.size)

    def generateEdges(self):
        generator = GraphGenerator(dataset=self.dataset, threshold=self.threshold)
        generator.generateEdges()

    def visualizeResults(self):
        ResultCollector(self.dataset).visualizeResults(self.type)
