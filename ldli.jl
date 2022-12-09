using LinearAlgebra
using SparseArrays
using Statistics
# using Pkg
# Pkg.build("Laplacians")
import Laplacians as Lap
# using CUDA
getLDLi(mat) = Lap.approxChol(Lap.LLmatp(mat))
getLDLiParallel(mat) = Lap.approxCholParallel(Lap.LLmatp(mat))
applyLDLi(ldli, mat) = sparse(reduce(hcat, map(x -> Lap.LDLsolver(ldli, Vector(x)), eachcol(mat))))

function run(n::Int)
        
    a = Lap.pure_random_graph(n)
    a = sparse([2, 4, 1, 3, 2, 4, 1, 3], [1, 1, 2, 2, 3, 3, 4, 4], [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0], 4, 4)
    display(a)
    la = Lap.lap(a)

    @time "getLDLi" ldli = getLDLi(deepcopy(a))
    print(ldli)
    # @time "getLDLiParrallel" ldlip = getLDLiParallel(deepcopy(a))
    # a

    # println("Lap:")
    # println("the max of deviation from identity matrix for all entries in la is $(maximum(broadcast(abs, (la - 1.0I))))")
    # println("the mean of deviation from identity matrix for all entries in la is $(mean(broadcast(abs, (la - 1.0I))))")
    # println()

    # res = applyLDLi(deepcopy(ldli), deepcopy(la))
    # println("ldli * la:")
    # println("the max of deviation from identity matrix for all entries is $(maximum(broadcast(abs, (res - 1.0I))))")
    # println("the mean of deviation from identity matrix for all entries is $(mean(broadcast(abs, (res - 1.0I))))")
    # # display(res[1:10, 1:10])
    # println()

    # res = applyLDLi(deepcopy(ldlip), deepcopy(la))
    # println("parallel ldli * la:")
    # println("the max of deviation from identity matrix for all entries is $(maximum(broadcast(abs, (res - 1.0I))))")
    # println("the mean of deviation from identity matrix for all entries is $(mean(broadcast(abs, (res - 1.0I))))")
    # # display(res[1:10, 1:10])
    # println()

    # println("the upper left corner of the resulting matrix is: ")
    # print(res[1:15, 1:15])
end


run(4)