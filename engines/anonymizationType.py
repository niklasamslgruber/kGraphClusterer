from enum import Enum, unique


@unique
class AnonymizationType(Enum):
    SaNGreeA = "SaNGreeA"
    DISCERNIBILITY = "Discernibility"
    PRECISION = "Precision"
    CLASSIFICATION_METRIC = "Classification_Metric"
    NORMALIZED_CERTAINTY_PENALTY = "Normalized_Certainty_Penalty"
    ENTROPY = "Entropy"
    MODULARITY = "Modularity"
    SILHOUETTE = "Silhouette"
    GRAPH_PERFORMANCE = "Graph_Performance"

    @staticmethod
    def getCase(method: str):
        match method:
            case AnonymizationType.SaNGreeA.name:
                return AnonymizationType.SaNGreeA
            case AnonymizationType.DISCERNIBILITY.name:
                return AnonymizationType.DISCERNIBILITY
            case AnonymizationType.PRECISION.name:
                return AnonymizationType.PRECISION
            case AnonymizationType.CLASSIFICATION_METRIC.name:
                return AnonymizationType.CLASSIFICATION_METRIC
            case AnonymizationType.NORMALIZED_CERTAINTY_PENALTY.name:
                return AnonymizationType.NORMALIZED_CERTAINTY_PENALTY
            case AnonymizationType.ENTROPY.name:
                return AnonymizationType.ENTROPY
            case AnonymizationType.MODULARITY.name:
                return AnonymizationType.MODULARITY
            case AnonymizationType.SILHOUETTE.name:
                return AnonymizationType.SILHOUETTE
            case AnonymizationType.GRAPH_PERFORMANCE.name:
                return AnonymizationType.GRAPH_PERFORMANCE
            case _:
                return
