from enum import Enum, unique


@unique
class AnonymizationType(Enum):
    SaNGreeA = "SaNGreeA"
    DISCERNIBILITY_ALL = "Discernibility-All"
    DISCERNIBILITY = "Discernibility"

