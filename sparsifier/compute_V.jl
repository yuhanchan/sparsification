using Pkg
ENV["PYTHON"] = "/home/chenyh/local/miniconda3/envs/pyg/bin/python"

# Pkg.build("PyCall")

# Pkg.add("DelimitedFiles")
# Pkg.add("NPZ")
# Pkg.add("Laplacians")
# Pkg.build("Laplacians")
# Pkg.add("PyCall")
# Pkg.add("ArgParse")
# Pkg.add("Profile")
# Pkg.add("PProf")
# Pkg.add("ProfileView")

# using ProfileView
using Profile
using PProf
using PyCall
using DelimitedFiles
using SparseArrays
using Laplacians
using LinearAlgebra
using ArgParse

np = pyimport("numpy");
scipy = pyimport("scipy");

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--filepath"
        help = "Path to the file to read"
        required = true
    end

    return parse_args(s)
end

const scipy_sparse_find = pyimport("scipy.sparse")["find"]
function mysparse(Apy::PyObject)
    IA, JA, SA = scipy_sparse_find(Apy)
    return sparse(Int[i + 1 for i in IA], Int[i + 1 for i in JA], SA)
end

function compute_V(a, JLfac=4.0)

    # print("Preparing approxchol_lap...")
    @time "approxchol_lap" f = approxchol_lap(a, tol=1e-2) # a is the weight matrix loaded from file, nxn
    # exit()

    n = size(a, 1)
    k = round(Int, JLfac * log(n)) # number of dims for JL, natrual log
    println("k = ", k)
    @time "compute (W^1/2)*B" U = wtedEdgeVertexMat(a) # equal to (W^1/2)*B in paper, mxn 
    println("dim of U: ", size(U))
    m = size(U, 1)

    println("Generate R (Q): ")
    R = @time "generate R (Q)" randn(Float64, m, k) # equal to Q, JL projection matrix, mxk

    print("Computing QW: ")
    UR = @time "compute Q(W^1/2)B" U' * R # nxk

    V = zeros(n, k)
    @time string("total time for k=", k, " iteration") Threads.@threads for i in 1:k
        print(i, "/", k)
        V[:, i] = @time string("time for i=", i) f(UR[:, i]) # f is the linear solver, solve for x in Ax=b
    end

    return V # V here is the Z in paper
end


# filenames = ["Cora","Citeseer","Pubmed","Phy","CS","Photo","Computers"]
# filenames=["Cora","Pubmed","Phy","CS"]
# filenames=["Photo","Computers"]
# filenames=["git","twitch_DE","twitch_FR","wiki_crocs","wiki_squirrels"]
# filenames=["Reddit_sparse"]
# filenames=["Reddit"]

# define the path to the adjacency matrix files in the next line and uncomment it

filepath = "./"

function main()
    parsed_args = parse_commandline()

    println("Loading file:... ")
    data = @time "load npz" scipy.sparse.load_npz(parsed_args["filepath"])
    println("Finished loading file, ")


    println("Converting into Julia format")
    B = @time "convert to julia's sparse format" mysparse(data)
    B = @time "convert to julia's sparse format" convert(SparseMatrixCSC{Float64,Int64}, B)
    println("Finished converting into Julia format")

    println("Computing V matrix..")
    # @profile V = @time "compute_V" compute_V(B)
    # @pprof V = compute_V(B)
    V = compute_V(B)
    println("Finished computing V matrix..")

    println("Dimensions of V are: ")
    println(size(V))
    println("saving V matrix:...")
    filename = string(rsplit(parsed_args["filepath"], ".", limit=2)[1], ".csv")
    print("Saving to file... ")
    @time "write to file" writedlm(filename, V, ',')
end

main()

