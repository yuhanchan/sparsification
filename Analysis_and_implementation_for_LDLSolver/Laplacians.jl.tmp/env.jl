include("src/Laplacians.jl")
using SparseArrays
import .Laplacians as LapM
import Random
import Laplacians as Lap

applyLDLi(ldli, mat) = sparse(reduce(hcat, map(x -> Lap.LDLsolver(ldli, Vector(x)), eachcol(mat))))
getLDLi(mat) = Lap.approxChol(Lap.LLmatp(mat))