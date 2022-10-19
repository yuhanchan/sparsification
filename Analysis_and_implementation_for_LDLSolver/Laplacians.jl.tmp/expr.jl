include("env.jl")

a = LapM.pure_random_graph(50)
llmatp = LapM.LLmatp(a)
llmatp2 = LapM.LLmatp(a)

Random.seed!(1234)
ldli = LapM.approxChol(llmatp)

Random.seed!(1234)
ldli2 = LapM.approxChol_alt(llmatp2)

println(ldli.col == ldli2.col)
println(ldli.colptr == ldli2.colptr)
println(ldli.rowval == ldli2.rowval)
println(isapprox(ldli.fval, ldli2.fval))
println(isapprox(ldli.d, ldli2.d))