import unittest
import pandas as pd
from engines.anonymizationEngine import AnonymizationEngine
from dataHandler.dataProcessor import DataProcessor
from dataHandler.datasets import Datasets
from engines.gilEngine import GILEngine
from models.graph import Graph
from models.partition import Partition


class NGILTests(unittest.TestCase):
    graph: Graph
    edges: pd.DataFrame
    features: pd.DataFrame
    numerical_identifiers: [str] = ["age"]
    categorical_identifiers: [str] = ["zip", "gender"]

    def __init__(self, *args, **kwargs):
        super(NGILTests, self).__init__(*args, **kwargs)
        self.edges = DataProcessor.loadEdges(Datasets.SAMPLE)
        self.features = DataProcessor.loadFeatures(Datasets.SAMPLE)
        self.graph = Graph().create(self.edges, self.features, self.numerical_identifiers, self.categorical_identifiers)

    def testGILCalculationForPartition1(self):
        partition1 = Partition.create(self.graph, [[4, 7, 8], [1, 2, 3], [5, 6, 9]])
        partition2 = Partition.create(self.graph, [[4, 5, 6], [1, 2, 3], [7, 8, 9]])
        gil_values = [7.73076923076923, 14.307692307692307]
        ngil_values = [0.2863247863247863, 0.5299145299145299]

        for (index, partition) in enumerate([partition1, partition2]):
            engine = GILEngine(self.graph, partition)
            gil = engine.getGraphGIL()
            ngil = engine.getNGIL()

            self.assertEqual(gil, gil_values[index])
            self.assertEqual(ngil, ngil_values[index])

    def testAnonymizerWithBetaZero(self):
        anonymizer = AnonymizationEngine(self.graph, 1, 0, 3)
        result: Partition = anonymizer.anonymize()
        self.assertEqual(len(result.clusters), 3)

        cluster_result = [{1, 2, 3}, {5, 6, 9}, {4, 7, 8}]
        real_result = list(map(lambda node: set(node.getIds()), result.clusters))

        for cluster in cluster_result:
            self.assertTrue(cluster in real_result)

    def testAnonymizerWithAlphaZero(self):
        anonymizer = AnonymizationEngine(self.graph, 0, 1, 3)
        result: Partition = anonymizer.anonymize()
        self.assertEqual(len(result.clusters), 3)

        cluster_result = [{1, 2, 3}, {4, 5, 6}, {7, 8, 9}]
        real_result = list(map(lambda node: set(node.getIds()), result.clusters))

        for cluster in cluster_result:
            self.assertTrue(cluster in real_result)




if __name__ == '__main__':
    unittest.main()
