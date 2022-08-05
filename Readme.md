### Dataset file extension convensions
``` bash
[dataset_name]
├── pruned                  # pruned version dir
│   ├── er                  # prune method dir
│   │   ├── 0.1             # prune rate dir
│   │   │   ├── duw.el      # directed unweighted edge list
│   │   │   └── dw.npy
│   │   └── ...
│   │   │   ├── ...
│   │   │   └── ...
│   │   └── 0.9
│   │       ├── duw.el      
│   │       └── dw.npy
│   ├── in_degree           # prune method dir
│   │   ├── 0.1
│   │   │   └── duw.el
│   │   ├── ...
│   │   │   └── ...
│   │   └── 0.9
│   │       └── duw.el
│   ├── out_degree          # prune method dir
│   │   ├── 0.1
│   │   │   └── duw.el
│   │   ├── ...
│   │   │   └── ...
│   │   └── 0.9
│   │       └── duw.el
│   └── random              # prune method dir
│       ├── 0.1
│       │   ├── duw.el
│       │   └── duw.el.ecg
│       ├── ...
│       │   ├── ...
│       │   └── ...
│       └── 0.9
│           ├── duw.el
│           └── duw.el.ecg
└── raw
    ├── duw.el
    ├── duw.el.ecg
    ├── duw.el.elm
    ├── duw.el.original
    ├── duw.el.original.symmetriced
    ├── duw.el.original.symmetriced.zerobased
    ├── duw.el.sorted
    ├── email-Eu-core-department-labels.txt
    ├── Reff.pkl
    ├── stage3.npz
    ├── uduw.el
    ├── V.csv
    └── V.npz
```

1. run pruning on sorted, eliminated, 0-based graphs, get pruned graphs
2. get the largest connected-components
3. make the graph 1-based
4. run node2vec
5. run CGE
