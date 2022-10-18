# Dependency
You should install the julia package if there misses one. There should be no other dependencies. The julia kernel I use is `julia-1.8.0`.

# Organization of this directory

## the `graph` directory
The `graph` directory should contain the edge list file from SNAP (with the comment of the head be removed). The scripts assumes that there exits "email-enron.txt", "soc-LiveJournal1.txt", "web-Google.txt", and "wiki-Talk.txt" datasets. Please check the notebook "Can we reorder?.ipynb". The second block of that notebook has a summary for the datasets being used.

## the `Laplacians.jl` directory
The `Laplacians.jl` directory contains a modified Laplacians library, whose implementation is closer to a parallelized version. This intermediate bridges the original serial code to parallelized implementation. I think there is no need to read the code here and `CUDA solver.ipynb` is the parallelized code.

## the root directory
Many notebooks reside here, each discusses a problem. A summary of each notebook will be listed below

# Summary of the notebooks

Not all notebooks are useful because some notebooks are some intermediate results. Followings are the notebook that contains useful data and plots.

## `Can we reorder?.ipynb`
This notebook consider can we fix the elimination order before ACD begins. It considers the limited parallelism for the original ACD as well as the effects of fixing the elimination order.

## `CUDA solver.ipynb`
This notebook includes the GPU accelerated LDLsolver, CPU & GPU combined LDLsolver, as well as a CPU baseline that is timed by parts. All code for the implemenation are in that notebook.

## `GPU-LDL-spMV.ipynb`
This notebook accelerate all parts except the LDLsolver in the PCG using CUDA library for sparse matrix-vector multiplication.

## `Graph Parallelism analyzer.ipynb`
This notebook analyzes the potential parallelism in the LDLsolver.

## `meaning.ipynb`
This notebook shows what `LDLsolver` and `LDLinv` means: a good preconditioner that approximatly makes the laplaician of a graph to identity.

## `Parellelism analysis for LDL solver.ipynb`
Another notebook talks about the potential parallelism. It also consider the dependency for the `forward!` is different from `backward!`.

## `Shape of forward.ipynb`
Discuss the meaning (in matrix) of the `forward!()` and `backward!()`. They are not triangular.
