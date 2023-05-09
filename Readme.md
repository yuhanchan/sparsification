## Above all
We may use the terms ``sparsify`` and ``prune`` interchangably in this doc

------
## Folder structure
``` bash
.
├── analysis # legacy code, do not use
├── config.json # config file for sparsification params
├── data # datasets raw files and pruned files
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




