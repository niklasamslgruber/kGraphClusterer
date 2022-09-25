from config import FLAGS
from runner import Runner

if __name__ == '__main__':
    print(f"Starting Clusterer MetricRun for {FLAGS.dataset}...\n")
    Runner().runMetrics()
