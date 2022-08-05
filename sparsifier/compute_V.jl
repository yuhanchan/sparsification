using Pkg
ENV["PYTHON"]="/home/chenyh/local/miniconda3/envs/py39/bin/python"

# Pkg.build("PyCall")

Pkg.add("DelimitedFiles")
Pkg.add("NPZ")
Pkg.add("Laplacians")
Pkg.add("PyCall")
Pkg.add("ArgParse")


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
    return sparse(Int[i+1 for i in IA], Int[i+1 for i in JA], SA)
end

function compute_V(a, JLfac=4.0)

  print("Preparing approxchol_lap...")
  f = @time approxchol_lap(a,tol=1e-2); # a is the weight matrix loaded from file, nxn

  n = size(a,1)
  k = round(Int, JLfac*log(n)) # number of dims for JL, natrual log
  println("k = ", k)
  U = wtedEdgeVertexMat(a) # equal to W, mxn
  println("dim of U: ", size(U))
  m = size(U,1)

  println("Generate R (Q): ")
  R = @time randn(Float64, m,k) # equal to Q, JL projection matrix, mxk

  print("Computing QW: ")
  UR = @time U'*R; # nxk

  V = zeros(n,k)
  @time Threads.@threads for i in 1:k
    print(i, "/", k)
    V[:,i] = @time f(UR[:,i]) # f is the linear solver, solve for x in Ax=b
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

filepath="./"

function main()
  parsed_args = parse_commandline()

  println("Loading file:... ")
  data = @time scipy.sparse.load_npz(parsed_args["filepath"]);
  println("Finished loading file, ")


  println("Converting into Julia format")
  B = @time mysparse(data);
  B = @time convert(SparseMatrixCSC{Float64,Int64},B);
  println("Finished converting into Julia format")

  println("Computing V matrix..")
  V = @time compute_V(B);
  println("Finished computing V matrix..")

  println("Dimensions of V are: ")
  println(size(V))
  println("saving V matrix:...")
  filename=string(rsplit(parsed_args["filepath"], ".", limit=2)[1], ".csv");
  print("Saving to file... ")
  @time writedlm(filename,  V, ',')
end

main()

