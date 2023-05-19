## Above all
We may use the terms ``sparsify`` and ``prune`` interchangably in this doc

------
## Folder structure
``` bash
.
├── analysis # legacy code, do not use
├── config.json # config file for sparsification params
├── data # datasets raw files and pruned files
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
├── profile.sh # legacy code, do not use
├── setup.py # setup file for this sparsifiction lib
├── sparsifier # code for ER sparsifier and some legacy sparsifiers
├── src
│   ├── legacy # folder containing legacy code, do not use
│   ├── Cheb.py  # code for running Cheb GNN
│   ├── ClusterGCN.py # code for running Cluseter GCN
│   ├── GCN.py # code for running GCN
│   ├── graph_reader.py # helper code for reading graphs
│   ├── logger.py # helpper code for logging GNN
│   ├── main.py # enter point
│   ├── metrics_gt.py # metric evaluaters using graph-tool
│   ├── metrics_nk.py # metric evaluaters using Networkit
│   └── sparsifiers.py # lib invoking all sparsifiers
├── tests # test files
├── utils # helper functions and binaries
└── workload # legacy workload code, do not use
```




## Env

Conda is recommendated to manage env. To install necessary packages:
1. Install conda. [link](https://docs.anaconda.com/free/anaconda/install/index.html)
2. Create an env named ``spar`` by running ``conda env create --file env.yaml``
3. Activate env by running ``conda activate spar``
4. Setup env by running ``source env.sh``. (Run step 3 and 4 every time a new terminal is started.)


## System Requirements
- OS: We run experiments on Ubuntu 20.04 LTS, with python 3.9.12. Other platform may also work, but not extensively tested. 
- Memory: Depends on the size of graphs.
- Storage: Graph size varies from ~MB to ~GB. However, to conduct end-to-end experiments for all sparsifiers, the storage required will quickly explode. Each graph will be sparsified using ``12`` sparsifiers, each with ``9`` different prune rates, and some non-deterministic sparsifiers will run ``3-10`` times to show variance. Also a directed/un-directed weighted/enweighted version (totaling ``4``) for each graph may be required for evaluating for some metrics. These factors altogether will leads to a ``100x-1000x`` storage expansion to only the original graph, and can quickly get to ``TB`` level. We recommend starting with small graphs first. 

## Dataset Download
run ``python utils/data_preparation.py [dataset_name/all]`` to download data. ``all`` will download all data, we recommend start with small datasets. datasets from small to large (by #edge) are *``(smallest) ego-Facebook, ca-HepPh, email-Enron, ca-AstroPh, com-Amazon, com-DBLP, web-NotreDame, ego-Twitter, web-Stanford, wiki-Talk, web-Google, web-BerkStan, human_gene2, ogbn-proteins, Reddit (largest)``*



