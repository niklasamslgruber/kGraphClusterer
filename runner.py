import itertools
import json
import pandas as pd
from config import FLAGS
from constants import RANDOM_SEED
from dataHandler.graphGenerator import GraphGenerator
from engines.anonymizationEngine import AnonymizationEngine
from dataHandler.dataProcessor import DataProcessor
from dataHandler.datasets import Datasets
from engines.anonymizationType import AnonymizationType
from engines.gilEngine import GILEngine
from engines.graphEvaluationEngine import GraphEvaluationEngine
from engines.resultCollector import ResultCollector
from engines.visualizationEngine import VisualizationEngine
from models.graph import Graph
import copy
import time

from models.partition import Partition


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

    @staticmethod
    def run(dataset: Datasets, alpha: float, beta: float, k: int, type: AnonymizationType, threshold: int,
            shouldPrint: bool = False):
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

        result = ResultCollector.Result(k, alpha, beta, len(features), len(edges), round(ngil, 4), round(nsil, 4),
                                        len(partition.clusters), exec_time, type.value,
                                        round(pow(pow(ngil - 0, 2) + pow((nsil * 100) - 0, 2), 0.5), 4))

        ResultCollector(dataset).saveResult(result)

        Runner.storeOutput(graph, partition, dataset, threshold, result)

        Runner.__verifyGraph(partition, k, threshold)

        if FLAGS.plot:
            visualizationEngine = VisualizationEngine(dataset, threshold)
            visualizationEngine.drawGraph(partition)
            visualizationEngine.drawInitialGraph()

        if shouldPrint:
            print("\n-----")
            print(f"Generated Clusters for dataset {dataset.name}:")
            for (index, cluster) in enumerate(partition.clusters):
                print(f"\tCluster {index + 1}:", cluster.getIds())
            print("\nStatistics:")
            print("\tNSIL:", nsil)
            print("\tNGIL:", ngil)
            print("Execution time", exec_time, "s")

    @staticmethod
    def storeOutput(graph: Graph, partition: Partition, dataset: Datasets, threshold, result):
        associations = set([])
        edges = set([])

        raw_features = pd.read_csv(dataset.getFeaturePath(), index_col="id").sample(threshold, random_state=RANDOM_SEED)
        matched_features = pd.read_csv(dataset.getAssociationPath(threshold))

        for cluster in partition.clusters:
            relations = list(
                itertools.chain.from_iterable(
                    list(map(lambda node_item: node_item.relations, cluster.nodes))))

            connected_clusters = set([])
            for relation in relations:
                clusters = set(list(filter(lambda cluster: relation in cluster.getIds(), partition.clusters)))
                for connected_cluster in clusters:
                    connected_clusters.add((cluster.id, connected_cluster.id))
                    edges.add((cluster.id, connected_cluster.id))

            generalized_value = GILEngine(graph, Partition([]), dataset).mergeNodes(cluster)
            for categorical_identifier in dataset.getCategoricalIdentifiers():
                generalized_value[categorical_identifier] = generalized_value[categorical_identifier][0]

            for node in cluster.nodes:
                associations.add((cluster.id, node.id))
                column = matched_features[matched_features["transactionID"] == node.id]["id"].tolist()[0]

                for identifier in dataset.getNumericalIdentifiers() + dataset.getCategoricalIdentifiers():
                    raw_features.at[column, identifier] = str(generalized_value[identifier])

                raw_features.at[column, "nodeId"] = str(node.id)

        edge_frame = pd.DataFrame(edges, columns=["id1", "id2"]).sort_values(by="id1")
        association_frame = pd.DataFrame(associations, columns=["clusterId", "nodeId"]).sort_values(by="clusterId")

        raw_features.set_index("nodeId", inplace=True)
        raw_features.to_csv(f"{dataset.getOutputPath()}/features.csv")
        edge_frame.to_csv(f"{dataset.getOutputPath()}/edges.csv", index=None)
        association_frame.to_csv(f"{dataset.getOutputPath()}/associations.csv", index=None)
        with open(f"{dataset.getOutputPath()}/results.json", 'w') as file:
            json.dump(result.to_dict(), file, sort_keys=True, indent=4)

    @staticmethod
    def __verifyGraph(partition: Partition, k: int, threshold: int):
        numberOfNodes = 0
        for cluster in partition.clusters:
            numberOfNodes += len(cluster.nodes)
            assert len(cluster.nodes) >= k
            for node in cluster.nodes:
                assert node.cluster_id is not None
                assert node.cluster_id == cluster.id, f"{node.cluster_id}{cluster.id}"
                assert sorted(node.relations) == sorted(list(set(node.relations)))
                assert node.id not in node.relations

        assert numberOfNodes == threshold, f"Number of nodes ({numberOfNodes}) is not equal to dataset size ({threshold})"

    def runMultiple(self):
        a_b_pairs = [(1, 0), (1, 0.5), (0.5, 1), (1, 1)]
        for k in [2, 4, 6, 8, 10]:
            for (alpha, beta) in a_b_pairs:
                for limit in [100, 300, 500, 1000]:
                    self.run(self.dataset, alpha, beta, k, self.type, limit)

    def runMetrics(self):
        for metric in AnonymizationType:
            self.run(self.dataset, FLAGS.alpha, FLAGS.beta, FLAGS.k, metric, FLAGS.size)

    def generateEdges(self):
        generator = GraphGenerator(dataset=self.dataset, threshold=self.threshold)
        generator.generateEdges()
