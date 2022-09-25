from enum import Enum, unique


@unique
class AnonymizationType(Enum):
    SaNGreeA = "SaNGreeA"
    DISCERNIBILITY = "Discernibility"
    PRECISION = "Precision"
    CLASSIFICATION_METRIC = "Classification_Metric"

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
            case _:
                return
