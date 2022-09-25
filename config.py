import argparse
from dataHandler.datasets import Datasets
from engines.anonymizationType import AnonymizationType

parser = argparse.ArgumentParser()

default_k = 3
parser.add_argument('-k', '--k', default=default_k, type=int, help=f"k-degree of anonymization (default: {default_k})", metavar='')

default_alpha = 0.5
parser.add_argument('-a', '--alpha', default=default_alpha, type=int, help=f"weight of generalization information loss (default: {default_alpha})", metavar='')

default_beta = 0.5
parser.add_argument('-b', '--beta', default=default_beta, type=int, help=f"weight of structural information loss (default: {default_beta})", metavar='')

default_size = 300
parser.add_argument('-n', '--size', default=default_size, type=int, help=f"dataset size (default: {default_size})", metavar='')

available_methods = [type.name for type in AnonymizationType]
parser.add_argument('-m', '--method', default=available_methods[0], type=str, help=f"information loss metric for anonymization (available: {', '.join(available_methods)}) (default: {available_methods[0]})", metavar='')

parser.add_argument('--plot', default=False, action='store_true', help="plot results")
parser.add_argument('--generate_edges', type=int, help=f"generate X edges based on the BTC transaction dataset", metavar='')

available_datasets = [Datasets.ADULTS.name, Datasets.BANK_CLIENTS.name]
parser.add_argument('--dataset', default=available_datasets[0], type=str, help=f"information loss metric for anonymization (available: {', '.join(available_datasets)}) (default: {available_datasets[0]})", metavar='')

FLAGS = parser.parse_args()
