## Above all
We may use the terms ``sparsify`` and ``prune`` interchangably in this doc

------
## Folder structure
``` bash
.
├── config.json # config file for sparsification params
├── data # datasets raw files and pruned files, auto-created
    ├──...
          ├── # graphs are stored in edgelist format.
              # uduw.el, duw.el, udw.wel, dw.wel means 
              # undirected-unweighted, directed-unweighted,
              # undirected-weighted, directed-weighted 
              # edgelist files, respectively.
├── dataLoader # code for loading datasets
├── env.sh # bash file for setting PROJECT_HOME
├── experiments # output folder for GNN, auto-created
├── myLogger # logger lib
├── output_metric_parsed # parsed metric evaluation output, auto-created when parse output_metric_raw
├── output_metric_plot # plot output, auto-created when run plot on output_metric_parsed
├── output_metric_raw # raw metric evaluation output, auto-created when run eval
├── output_sparsifier_parsed # parsed sparsifier output, auto-created when parse output_sparsifier_raw
├── output_sparsifier_raw # raw sparsifier evaluation output, auto-created when run sparsify
├── paper_fig # reproduced fig same as the ones in the paper
├── parser # parser code for parsing raw outputs and generate parsed otuput
├── plot # ploter code
├── setup.py # setup file for this sparsifiction lib
├── sparsifier # code for ER sparsifier and some legacy sparsifiers
├── src
│   ├── Cheb.py  # code for running Cheb GNN
│   ├── ClusterGCN.py # code for running Cluseter GCN
│   ├── GCN.py # code for running GCN
│   ├── graph_reader.py # helper code for reading graphs
│   ├── logger.py # helpper code for logging GNN
│   ├── main.py # enter point
│   ├── metrics_gt.py # metric evaluaters using graph-tool
│   ├── metrics_nk.py # metric evaluaters using Networkit
│   └── sparsifiers.py # lib invoking all sparsifiers
└── utils # helper functions and binaries
```




## Env

Conda is recommendated to manage env. To install necessary packages:
1. Install conda following [link](https://docs.anaconda.com/free/anaconda/install/index.html).
2. Create an env named ``spar`` by running ``conda env create --file env.yaml``
3. Activate env by running ``conda activate spar``
4. Install extra packages by running ``pip install networkit setuptools==52.0.0``. Latest version of setuptool cause weird errors.
5. Install the current folder by running ``pip install -e .`` (make sure you type the dot (``.``) in the command)
6. Install ``torch`` following [link](https://pytorch.org/get-started/locally/).
7. Install ``PyG`` following [link](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
8. Install additional packages by running ``pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html``, where ``${TORCH}`` and ``${CUDA}`` is the torch and cuda version you use, follow [link](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for more details.
9. Setup env by running ``source env.sh``. (Run step 3 and 9 every time a new terminal is started.)


## System Requirements
- OS: We run experiments on Ubuntu 20.04 LTS, with python 3.9.12. Other platform may also work, but not extensively tested. 
- Memory: Depends on the size of graphs.
- Storage: Graph size varies from ~MB to ~GB. However, to conduct end-to-end experiments for all sparsifiers, the storage required will quickly explode. Each graph will be sparsified using ``12`` sparsifiers, each with ``9`` different prune rates, and some non-deterministic sparsifiers will run ``3-10`` times to show variance. Also a directed/un-directed weighted/enweighted version (totaling ``4``) for each graph may be required for evaluating for some metrics. These factors altogether will leads to a ``100x-1000x`` storage expansion to only the original graph, and can quickly get to ``TB`` level. We recommend starting with small graphs first. 


## Compile util code
```bash
cd $PROJECT_HOME/utils
mkdir bin
make
cd $PROJECT_HOME
```


## Dataset Download and Pre-processing
```bash
python $PROJECT_HOME/utils/data_preparation.py --dataset_name [dataset_name/all]
```

This will download data and do necessary data pre-processin. ``all`` will download all data, we recommend start with small datasets. datasets from small to large (by #edge) are *``(smallest) ego-Facebook, ca-HepPh, email-Enron, ca-AstroPh, com-Amazon, com-DBLP, web-NotreDame, ego-Twitter, web-Stanford, wiki-Talk, web-Google, web-BerkStan, human_gene2, ogbn-proteins, Reddit (largest)``*


## Run
```bash
python $PROJECT_HOME/src/main.py --dataset_name [dataset_name/all] --mode [sparsify/eval/all/clean]
```

``--dataset_name`` indicates the dataset to use, use name instead of the dataset path, ``all`` will run for all datasets. It is recommended not to use ``all`` unless you know what you are doing because it can take a long time and large file space.

``--mode`` indicates what to run. ``sparsify`` will run all sparsifiers on the given ``dataset_name``. ``eval`` assumed the sparsified files already exists, and evaluate the performance of the sparsified graphs on all metrics, run ``eval`` only if you have run ``sparsify`` on the given dataset. ``all`` will run ``sparsify`` and ``eval`` in tandem. ``clean`` will delete all files (raw graph, sparsified graphs, metric output, sparsifier output) associated with given dataset_name.

To run in a finer granularity, e.g. if want to run only a subset of sparsifiers and/or a subset of evaluation metrics, you need to modify the ``$PROJECT_HOME/src/main.py`` file, simply comment out lines for specific sparsifers and evaluation metrics should do.

By default, profiling for sparsifiers are not enabled, to enable profiling, go to ``$PROJECT_HOME/sparsifiers.py`` and enable @profile decorator before each sparsifier function. In the meantime, make sure ``graphSparsifier()`` in ``main.py`` is called with ``multi_process=False``, or it will fail.


## Output
The log for running ``sparsify`` will be in ``$PROJECT_HOME/output_sparsifier_raw/[dataset_name].txt``, and the log for running ``eval`` will be in ``$PROJECT_HOME/output_metric_raw/[dataset_name]/[metric]/log``

## Output Parse
Two scripts are provided for raw output parse, ``$PROJECT_HOME/parser/sparsifier_parse.py``, and ``$PROJECT_HOME/parser/metric_parse.py``.

```bash
python $PROJECT_HOME/parser/{sparsifier_parse.py, metric_parse.py} --dataset_name [dataset_name]/all
```

As always, ``all`` will run for all datasets.

## Plot
Two script are provided for plotting parsed data, ``$PROJECT_HOME/plot/plot.py`` and ``$PROJECT_HOME/plot/paper_plot.py``

```bash
python $PROJECT_HOME/plot/plot.py --dataset_name [dataset_name]/all --metric [metric]/all
```
 will plot the specified dataset for the specified metric for all sparsifiers and prune rates. 

```bash
python $PROJECT_HOME/plot/paper_plot.py
``` 
will reproduce the figures used in the paper.