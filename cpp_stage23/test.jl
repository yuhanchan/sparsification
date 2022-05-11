# using Pkg

# Pkg.build("PyCall")

# Pkg.add("DelimitedFiles")
# Pkg.add("Laplacians")

using SparseArrays
using Laplacians
using LinearAlgebra
using DelimitedFiles
using ArgParse
using Printf
Base.show(io::IO, f::Float64) = @printf io "%1.5f" f
# using Formatting
#
#
# using Debugger
function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table s begin
    "--filepath"
      help = "Path to the file to read"
      required = true
  end
  
  return parse_args(s)
end 

function compute_V(a, JLfac=4.0)

    f = approxchol_lap(a,tol=1e-2); # a is the weight matrix loaded from file, nxn

    n = size(a,1)
    k = round(Int, JLfac*log(n)) # number of dims for JL, natrual log
    println("k = ", k)
    U = wtedEdgeVertexMat(a) # equal to W, mxn
    m = size(U,1)

    # R = randn(Float64, m,k) # equal to Q, JL projection matrix, mxk
    fin = open("/data3/chenyh/sparsification/cpp_stage23/normal_random.txt", "r")
    R = zeros(Float64, m,k)
    for j=1:k
        for i=1:m
            # R[i,j] = 0.05 * i + 0.01 * j
            R[i, j] = parse(Float64, readline(fin))
        end
    end
    close(fin)

    UR = U'*R; # nxk
    # println("U = \n", U)
    println("U' = ")
    for i=1:n
        println(U[i,:])
    end
    println("")

    println("R = ")
    for i=1:m
        println(R[i,:])
    end
    println("")

    println("UR = ")
    for i=1:n
        println(UR[i,:])
    end
    println("")

    V = zeros(n,k)
    for i in 1:k
        V[:,i] = f(UR[:,i]) # f is the linear solver, solve for x in Ax=b
    end

    println("Z = ")
    for i in 1:k
        println(V[:,i])
    end
    return V # V here is the Z in paper
end



function main()
    # a = SparseMatrixCSC(Int64(3), Int64(3), Int64[1, 3, 5, 7], Int64[2, 3, 1, 3, 1, 2], Float64[1, 2, 1, 5, 2, 5])
    # a = SparseMatrixCSC(Int64(6), Int64(6), Int64[1, 4, 8, 11, 14, 17, 19], Int64[2, 3, 5, 1, 3, 5, 6, 1, 2, 4, 3, 5, 6, 1, 2, 4, 2, 4], Float64[2, 3, 1, 2, 5, 7, 8, 3, 5, 2.1, 2.1, 7.7, 9.2, 1, 7, 7.7, 8, 9.2])
    # println("a = \n", a)

    # read in a from mat1.in
    parsed_args = parse_commandline()

    coo = readdlm(parsed_args["filepath"], Float64)

    # remove first row
    row, col, nnz = Array{Int64}(coo[1, 1:3])
    coo = coo[2:size(coo, 1),:]
    println("coo = \n", coo)

    # transpose a   
    coo = coo'

    row_idx = Array{Int64}(coo[1,:])
    row_idx = row_idx .+ 1
    col_idx = Array{Int64}(coo[2,:])
    col_idx = col_idx .+ 1
    val = Array{Float64}(coo[3,:])
    col_ptr = zeros(Int64, col+1)
    for i=1:size(col_idx, 1)
        col_ptr[col_idx[i]+1] += 1
    end
    col_ptr[1] += 1
    for i=2:size(col_ptr, 1)
        col_ptr[i] = col_ptr[i-1] + col_ptr[i]
    end

    # println("row_idx = \n", row_idx)
    # println("col_idx = \n", col_idx)
    # println("val = \n", val)
    # println("col_ptr = \n", col_ptr)


    a = SparseMatrixCSC(row, col, col_ptr, row_idx, val)
    println("a = \n", a)
    println("")
        
    # flips = flipIndex(a)
    # println("flips:", flips)

    V = compute_V(a);
    filename=string("Z.csv");
    println("Saving to file... ", filename)
    writedlm(filename,  V, ',')
end

main()

