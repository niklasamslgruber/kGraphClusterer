from config import FLAGS
from dataHandler.datasets import Datasets
from engines.anonymizationType import AnonymizationType
from engines.resultCollector import ResultCollector
from engines.visualizationEngine import VisualizationEngine
from runner import Runner

def verify():
    results = ResultCollector(Datasets.ADULTS).loadResults()

    missing = []
    a_b_pairs = [(1, 0), (1, 0.5), (0.5, 1), (1, 1)]

    for k in [2, 4, 6, 8, 10]:
        for (alpha, beta) in a_b_pairs:
            for limit in [100, 300, 500, 1000]:
                for type in AnonymizationType:
                    if limit == 1000 and type in [AnonymizationType.MODULARITY, AnonymizationType.GRAPH_PERFORMANCE,
                                                  AnonymizationType.SILHOUETTE]:
                        continue
                    x = results[
                        (results["k"] == k) &
                        (results["alpha"] == alpha) &
                        (results["beta"] == beta) &
                        (results["size"] == limit) &
                        (results["method"] == type.value)
                        ]

                    if x.empty:
                        missing.append((k, alpha, beta, limit, type.value))

    print(len(missing), missing)

def createTable(k: int, size: int):
    results = ResultCollector(Datasets.ADULTS).loadResults()
    a_b_pairs = [(1, 0), (1, 0.5), (0.5, 1), (1, 1)]
    lines = []
    for k in [k]:
        for (alpha, beta) in a_b_pairs:
            for limit in [size]:
                x = results[
                    (results["k"] == k) &
                    (results["alpha"] == alpha) &
                    (results["beta"] == beta) &
                    (results["size"] == limit)
                    ]

                frame = x[x["method"] == AnonymizationType.SaNGreeA.value]
                if not frame.empty:
                    dict: {str: str} = {}
                    base = frame["ngil"].tolist()[0]
                    min = round(x["ngil"].min(), 3)
                    v = False
                    for (index, row) in x.iterrows():
                        if row["method"] not in [AnonymizationType.PURITY.value, AnonymizationType.SILHOUETTE.value,
                                                 AnonymizationType.MODULARITY.value,
                                                 AnonymizationType.GRAPH_PERFORMANCE.value]:
                            dict[row["method"]] = (round(row["ngil"], 3), round(((row["ngil"] - base) / base) * 100, 2),
                                                   "AAAAA" if row["method"] == AnonymizationType.SaNGreeA.value else
                                                   row["method"], row["method"])
                            print(row["method"], alpha, beta, base, row["ngil"],
                                  round(((row["ngil"] - base) / base) * 100, 2), round(base - row["ngil"], 2))

                    dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1][2])}
                    backslash = "\\"
                    lines.append(
                        f"\multirow{{2}}*{{\\footnotesize({float(alpha)}, {float(beta)})}} & {' & '.join(list(map(lambda x: (backslash + 'textbf{' + '{:,.3f}'.format(dict[x][0]) + '}' if dict[x][0] == min else '{:,.3f}'.format(dict[x][0])) + (dict[x][3][0] if v else ''), dict)))}\\\\")
                    lines.append(f"& " + ' & '.join(list(map(lambda x: '\\footnotesize' + (
                        backslash + 'textbf{' + '{:,.2f}'.format(dict[x][1]) + '\\%}' if dict[x][
                                                                                             0] == min else '{:,.2f}\\%'.format(
                            dict[x][1])) + (dict[x][3][0] if v else ''), dict))) + "\\\\\\hline")

    print("\n".join(lines))


def visualize():
    VisualizationEngine(Datasets.ADULTS, 100).plotPerformance()
    VisualizationEngine(Datasets.ADULTS, 100).plotNGIL()
    VisualizationEngine(Datasets.ADULTS, 100).plotNSIL()


if __name__ == '__main__':
    print("Starting Clusterer...\n")
    print("Params:", ', '.join(f'{k}={v}' for k, v in vars(FLAGS).items()))

    verify()





