# using Pkg
# Pkg.build("Laplacians")
# using Laplacians
using LinearAlgebra
using SparseArrays
using Metis
using Statistics
import Laplacians as Lap

getLDLi(mat) = Lap.approxChol(Lap.LLmatp(mat))
applyLDLi(ldli, mat) = sparse(reduce(hcat, map(x -> Lap.LDLsolver(ldli, Vector(x)), eachcol(mat))))

function getLDLi_partitioned(g, npart::Int)
    # create partitions
    parts = Metis.partition(g, npart)

    # create permute based on partition
    perm = collect(range(start=1, step=1, stop=g.n))
    perm = perm[sortperm(parts)]

    # apply permute to graph
    g = g[perm, perm]
    println("g permuted:")
    display(g)

    # split subgraphs
    part_count = [[i, count(==(i), parts)] for i in sort(unique(parts))]
    subgraphs = Vector{SparseMatrixCSC}()

    for i in range(start=1, step=1, stop=length(part_count) - 1)
        part_count[i+1][2] = part_count[i][2] + part_count[i+1][2] # compute prefix sum
    end
    println(part_count)

    subg = SparseMatrixCSC(Matrix(g)[1:part_count[1][2], 1:part_count[1][2]])
    push!(subgraphs, subg)
    for i in range(start=1, step=1, stop=length(part_count) - 1)
        subg = SparseMatrixCSC(Matrix(g)[part_count[i][2]+1:part_count[i+1][2], part_count[i][2]+1:part_count[i+1][2]])
        push!(subgraphs, subg)
    end

    # getLDLi for each subgraph and merge them
    ldli = Lap.LDLinv(g)
    ldli.col = Int64[]
    ldli.colptr = Int64[]
    ldli.d = Int64[]
    col_adjust = 0
    colptr_adjust = 0
    rowval_adjust = 0
    for subg in subgraphs
        display(subg)
        subldli = getLDLi(subg)
        # println(subldli)
        subldli.col .+= col_adjust
        col_adjust += subg.n - 1
        subldli.colptr .+= colptr_adjust
        colptr_adjust = last(subldli.colptr) - 1
        subldli.rowval .+= rowval_adjust
        rowval_adjust += subg.n

        append!(ldli.col, subldli.col)
        append!(ldli.colptr, subldli.colptr)
        append!(ldli.rowval, subldli.rowval)
        append!(ldli.fval, subldli.fval)
        append!(ldli.d, subldli.d)
    end
    return g, ldli
end

n = 1000

g = Lap.pure_random_graph(n)
println("g:")
display(g)

la = Lap.lap(g)
la_pd = la + I(n) * 0.01 # add a small number to diagonal to make it postive definite


# perm, iperm = Metis.permutation(g)
# display(g[perm, perm])

# C = LinearAlgebra.cholesky(la_pd)
# L = Matrix(sparse(C.L))
# display(L)

# display(L*L'-la[C.p, C.p])
# # println(L*L'≈la[C.p, C.p])

# res = applyLDLi(deepcopy(ldli), deepcopy(la))
# display(res)


# cg and pcg time comparison
v = rand(Float64, n)
v = v .- mean(v)

Lap.cg(la, v, verbose=true)

ldli = getLDLi(deepcopy(g))
pre(b) = Lap.LDLsolver(ldli, b)
Lap.pcg(la, v, pre, verbose=true)

g_permuted, ldli = getLDLi_partitioned(g, 4)
# println(ldli)
pre(b) = Lap.LDLsolver(ldli, b)
la = Lap.lap(g_permuted)
Lap.pcg(la, v, pre, verbose=true)

# getLDLi_partitioned(g, 4)




