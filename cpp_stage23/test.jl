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

const PRINT_MATRIX = false
const FILE_TYPE = 1 # 1: with weight and headings, 2: without weight and headings

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

    if PRINT_MATRIX
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
    end

    V = zeros(n,k)
    @time for i in 1:k
        @time V[:,i] = f(UR[:,i]) # f is the linear solver, solve for x in Ax=b
    end

    if PRINT_MATRIX
        println("Z = ")
        for i in 1:k
            println(V[:,i])
        end
    end

    return V # V here is the Z in paper
end



function main()

    # read in a from mat1.in
    parsed_args = parse_commandline()

    if FILE_TYPE == 1
        coo = readdlm(parsed_args["filepath"], Float64)

        # remove first row
        row, col, nnz = Array{Int64}(coo[1, 1:3])
        coo = coo[2:size(coo, 1),:]

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
    elseif FILE_TYPE == 2
        coo = readdlm(parsed_args["filepath"], Float64)

        coo = coo'

        row_idx = Array{Int64}(coo[2,:])
        row_idx = row_idx .+ 1
        col_idx = Array{Int64}(coo[1,:])
        col_idx = col_idx .+ 1

        row = maximum(row_idx)
        col = maximum(col_idx)
        row = max(row, col)
        col = max(row, col)
        nnz = size(coo, 2)

        # val should all be 1
        val = ones(Float64, nnz)

        col_ptr = zeros(Int64, col+1)
        for i=1:size(col_idx, 1)
            col_ptr[col_idx[i]+1] += 1
        end
        col_ptr[1] += 1
        for i=2:size(col_ptr, 1)
            col_ptr[i] = col_ptr[i-1] + col_ptr[i]
        end
    else
        println("FILE_TYPE is not 1 or 2")
        exit(1)
    end

    # println("row_idx = \n", row_idx[1:10])
    # println("col_idx = \n", col_idx[1:10])
    # println("val = \n", val[1:10])
    # println("col_ptr = \n", col_ptr[1:10])

    # exit(0)

    a = SparseMatrixCSC(row, col, col_ptr, row_idx, val)

    if PRINT_MATRIX
        println("a = \n", a)
        println("")
    end

    lap_a = lap(a)

    if PRINT_MATRIX
        println("lap_a = \n", lap_a)
        println("")
    end
        
    # flips = flipIndex(a)
    # println("flips:", flips)

    V = compute_V(a);
    filename=string("Z.csv");
    println("Saving to file... ", filename)
    writedlm(filename,  V, ',')
end

main()

