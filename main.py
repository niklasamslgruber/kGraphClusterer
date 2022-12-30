from config import FLAGS
from dataHandler.datasets import Datasets
from engines.anonymizationType import AnonymizationType
from engines.resultCollector import ResultCollector
from engines.visualizationEngine import VisualizationEngine
from runner import Runner


def createTable(k: int, size: int, ngil: bool):
    metric = "ngil" if ngil else "nsil"
    ROUND = 3 if ngil else 5
    results = ResultCollector(Datasets.BANK_CLIENTS).loadResults()
    a_b_pairs = [(1, 0), (1, 0.5), (0.5, 1)]
    lines = []
    for k in [k]:
        for (alpha, beta) in a_b_pairs:
            for limit in [size]:
                x = results[
                    (results["k"] == k) &
                    (results["alpha"] == alpha) &
                    (results["beta"] == beta) &
                    (results["size"] == limit)
                    &
                    (results["method"].isin([AnonymizationType.PURITY.value, AnonymizationType.SILHOUETTE.value,
                                                 AnonymizationType.MODULARITY.value,
                                                 AnonymizationType.GRAPH_PERFORMANCE.value]) == False)
                    ]
                frame = x[x["method"] == AnonymizationType.SaNGreeA.value]
                if not frame.empty:
                    dict: {str: str} = {}
                    base = frame[metric].tolist()[0]
                    min = round(x[metric].min(), ROUND)
                    v = False
                    for (index, row) in x.iterrows():
                        if True:
                            value = round(((row[metric] - base) / base) * 100, 2) if base > 0 else 0
                            dict[row["method"]] = (round(row[metric], ROUND), value,
                                                   "AAAAA" if row["method"] == AnonymizationType.SaNGreeA.value else
                                                   row["method"], row["method"])

                    dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1][2])}
                    backslash = "\\"
                    format = '{:,.3f}' if ngil else '{:,.5f}'
                    multirow = "\multirow{2}" if ngil else "\multirow{1}"
                    lines.append(
                        f"{multirow }*{{\\footnotesize({float(alpha)}, {float(beta)})}} & {' & '.join(list(map(lambda x: (backslash + 'textbf{' + format.format(dict[x][0]) + '}' if dict[x][0] == min else '{:,.3f}'.format(dict[x][0])) + (dict[x][3][0] if v else ''), dict)))}\\\\" + ("\\hline" if not ngil else ""))
                    if ngil:
                        lines.append(f"& " + ' & '.join(list(map(lambda x: '\\footnotesize' + (
                            backslash + 'textbf{' + '{:,.2f}'.format(dict[x][1]) + '\\%}' if dict[x][
                                                                                                 0] == min else '{:,.2f}\\%'.format(
                                dict[x][1])) + (dict[x][3][0] if v else ''), dict))) + "\\\\\\hline")

    print("\n".join(lines))

def createMinimumTable(ngil: bool):
    metric = "cgil"
    results = ResultCollector(Datasets.BANK_CLIENTS).loadResults()
    lines = []
    for size in [100, 300, 500, 1000]:
        dict: {int: str} = {}
        for k in [2, 4, 6, 8, 10]:
            x = results[(results["k"] == k) & (results["size"] == size)]
            minimum = x[metric].min()
            frame = x[x[metric] == minimum].iloc[0]
            if k == 8 and size == 100:
                print(frame, "X")

            if not frame.empty:

                match frame["method"]:
                    case AnonymizationType.SaNGreeA.value:
                        value = "Base"
                    case AnonymizationType.DISCERNIBILITY.value:
                        value = "Disc"
                    case AnonymizationType.PRECISION.value:
                        value = "Prec"
                    case AnonymizationType.CLASSIFICATION_METRIC.value:
                        value = "CM"
                    case AnonymizationType.NORMALIZED_CERTAINTY_PENALTY.value:
                        value = "NPC"
                    case AnonymizationType.ENTROPY.value:
                        value = "Entropy"
                    case _:
                        print(frame["method"])
                        value = frame["method"]

                print(value, k)
                dict[frame["k"]] = (round(frame[metric], 3), value, frame["alpha"], frame["beta"], k)

        dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1][4])}
        print(dict)
        backslash = "\\"
        multirow = "\multirow{2}" if ngil else "\multirow{1}"
        lines.append(f"{multirow}*{{{size}}} & {' & '.join(list(map(lambda x: backslash + 'footnotesize ' + dict[x][1], dict)))}\\\\")
        lines.append(f"& " + ' & '.join(list(map(lambda x: '\\footnotesize ' + ('$(' + str(dict[x][2]) + "," + str(dict[x][3]) + ')$'), dict))) + "\\\\\\hline")

    print("\n".join(lines))


def visualize():
    VisualizationEngine(Datasets.ADULTS, 100).plotPerformance()
    VisualizationEngine(Datasets.ADULTS, 100).plotNGIL()
    VisualizationEngine(Datasets.ADULTS, 100).plotNSIL()
    VisualizationEngine(Datasets.ADULTS, 100).plotCGIL()


if __name__ == '__main__':
    print("Starting Clusterer...\n")
    print("Params:", ', '.join(f'{k}={v}' for k, v in vars(FLAGS).items()))

    Runner().start()



