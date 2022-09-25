from config import FLAGS
from runner import Runner

if __name__ == '__main__':
    print("Starting Clusterer...\n")
    print("Params:", ', '.join(f'{k}={v}' for k, v in vars(FLAGS).items()))
    Runner().start()
