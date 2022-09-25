from enum import Enum, unique


@unique
class AnonymizationType(Enum):
    SaNGreeA = "SaNGreeA"
    DISCERNIBILITY_ALL = "Discernibility-All"
    DISCERNIBILITY = "Discernibility"
    PRECISION = "Precision"

    @staticmethod
    def getCase(method: str):
        match method:
            case AnonymizationType.SaNGreeA.name:
                return AnonymizationType.SaNGreeA
            case AnonymizationType.DISCERNIBILITY_ALL.name:
                return AnonymizationType.DISCERNIBILITY_ALL
            case AnonymizationType.DISCERNIBILITY.name:
                return AnonymizationType.DISCERNIBILITY
            case AnonymizationType.PRECISION.name:
                return AnonymizationType.PRECISION
            case _:
                return
