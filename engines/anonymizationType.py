from enum import Enum, unique


@unique
class AnonymizationType(Enum):
    SaNGreeA = "SaNGreeA"
    DISCERNIBILITY = "Discernibility"
    PRECISION = "Precision"

    @staticmethod
    def getCase(method: str):
        match method:
            case AnonymizationType.SaNGreeA.name:
                return AnonymizationType.SaNGreeA
            case AnonymizationType.DISCERNIBILITY.name:
                return AnonymizationType.DISCERNIBILITY
            case AnonymizationType.PRECISION.name:
                return AnonymizationType.PRECISION
            case _:
                return
