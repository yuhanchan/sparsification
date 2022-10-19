using Laplacians
spmat = pure_random_graph(50000)
display(spmat)
# println(spmat.m)
# println(spmat.n)
# println(spmat.colptr)
# println(spmat.rowval)
# println(spmat.nzval)
flips = Laplacians.flipIndex(spmat)
# println(flips)

open("small.bin", "w") do file
  write(file, spmat.m)
  write(file, spmat.n)
  write(file, spmat.colptr)
  write(file, spmat.rowval)
  write(file, spmat.nzval)
  write(file, flips)
end