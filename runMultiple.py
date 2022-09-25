from config import FLAGS
from runner import Runner

if __name__ == '__main__':
    print(f"Starting Clusterer MultiRun for {FLAGS.dataset} with method {FLAGS.method}...\n")
    Runner().runMultiple()
