# kGraphClusterer

`kGraphClusterer` extends the [SaNGreeA](https://api.semanticscholar.org/CorpusID:14209843) (_Social Network Greedy Anonymization_) algorithm by integrating various information loss metrics that influence the anonymization process.

## Description
This algorithm achieves k-Anonymity for any graph dataset by clustering datapoints together and reporting only the generalized cluster information instead of the single datapoints.

### Supported Information Loss Metrics
Nodes from a graph are assigned to the cluster where the information loss is the smallest compared to all other clusters. The information loss can be calculated with various metrics. The metric can be selected with the `--method` argument. Their weight can be controlled using the `--alpha` argument.

The currently supported metrics are:
* `SanGreeA`
* `DISCERNIBILITY`
* `PRECISION`
* `CLASSIFICATION_METRIC`
* `NORMALIZED_CERTAINTY_PENALTY`

## Running
```
$ python main.py
```

###### Optional Arguments: 
* `-h, --help`: See help including all default values and options
* `-k, --k` _(int)_: k-degree of the anonymization 
* `-a, --alpha`_(float)_: Weight  factor of generalization information loss
* `-b, --beta`_(float)_: Weight factor of structural information loss
* `-n, --size`_(int)_: Dataset subset size
* `-m, --method`_(str)_: Method for calculating information loss (_see above for options. Case-sensitive_)
* `--plot`: Plot all algorithm results
* `--generate_edges`_(int)_: Generate X edges based on the BTC transaction dataset
* `--dataset`_(str)_: Select dataset

##### Other Options
* Run the algorithm once for all available metrics. Parameters can be set using the command line arguments. If none are specified, the default values will be used.

	```
	$ python runMetrics.py
	```
	
* Run the algorithm for multiple parameter (alpha, beta, k, size) variations (set `--method` flag to specify the information loss metric)

	```
	$ python runMultiple.py
	```
	
##### Output
The anonymized graph will be stored in the `/output/<experiment_time>` directory of your dataset. It included four files that can be important to any graph database.
These files are:
* `associations.csv`: Includes all node IDs and the cluster ID they belong to
* `edges.csv`: Edges between the clusters
* `features.csv`: The anonymized node labels
* `results.json`: All experiment parameters

## Dependencies
The project requires Python `3.10` and the following packages need to be installed:
* `pandas`
* `tqdm`
* `matplotlib`
* `networkx`

> It is advised to create a separate Python environment using `Anaconda` or equivalent.

## Datasets
The repository includes two datasets:
* [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) (n = 45211)
* [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult) (n = 48842)

> Note: Running the algorithm for the full dataset is not advised due to the inefficient performance of the underlying greedy algorithm. Use `--size` to specify a smaller subset.

### Graph Conversion
The algorithm requires a graph dataset with a `features.csv` and `edges_{size}.csv` file. The `size` parameter must match the used `--size` argument.
This repository includes various edge files for both datasets (node size: 100, 300, 500 and 1,000). To generate edge files of other sizes for the default datasets run:

 ```
 python main.py --generate_edges <NUMBER_OF_NODES>
 ```

The default datasets use the [Harvard Bitcoin Transaction (2019-10-03)](https://dataverse.harvard.edu/file.xhtml?fileId=5635412&version=1.0) graph as a basis for the edge generation. This is because both default datasets are relational. Thus, each BTC address from the Harvard graph is mapped randomly to one row from the datasets.

### Custom Datasets
To use your own dataset create a new directory in `data/` with the name of the dataset. Additionally, you need to add the following files:
* **Edges**: Edge files should be located in a `edge` directory and should be named `edges_{size}.csv` . The files should have no header or index, just two columns with a node identifier each. 
	> Additionally, edges **must** be symmetric, i.e. an edge between node A and node B must exist twice as (A, B) and (B, A) in the edge file
* **Features**: A `features.csv` file should exist in the `data` directory which can include any data and represents the label information of each node. The file must include headers.
> The header must include a `id` column to uniquely identify each node. The IDs should match the ones from the edge files.
* **Generalization Trees**: Each categorical quasi-identifier must have a `{identifier_name}_generalization_tree.json` file which represents the generalization hierarchy. The top key of the JSON file must be the value of the highest generalization degree. 
	Example:

	```json
	{
	  "*****": {
	    "482**": [
	      "48201"
	    ],
	    "410**": [
	      "41076",
	      "41088"
	    ]
	  }
	}
	```

* **Register**: To register the custom dataset, add a new enum case in `datasets.py` and set the enum value to the name of your dataset. Add the new case in the `getCase()` function below.
* **Associations**: If your dataset should be based on the Harvard BTC dataset, simply run the algorithm with `python main.py` and a random association file will be created for you. If you want to add your own association files, add them in a `associations` directory named as `associations_{size}.csv`. The file must have two headers `id` and `transactionID` which maps the nodes from the `features.csv` to the edge identifiers in the edge files.