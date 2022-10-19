

@test L.path_graph_ijv(1) == L.empty_graph_ijv(1)
@test path_graph(1) == empty_graph(1)

@test L.hash(L.path_graph_ijv(1)) == L.hash(L.empty_graph_ijv(1))
L.hash(L.path_graph_ijv(5), hash(1))

L.transpose(L.path_graph_ijv(5))
L.adjoint(L.path_graph_ijv(5))
L.compress(L.path_graph_ijv(5))

@test L.star_graph(1) == L.empty_graph(1)
@test L.star_graph(4) == L.sparse([0 1 1 1 ; 1 0 0 0 ; 1 0 0 0 ; 1 0 0 0 ])
@test L.star_graph_ijv(1) == L.empty_graph_ijv(1)
@test L.star_graph_ijv(4) == Laplacians.IJV{Float64,Int64}(4, 6, [1, 1, 1, 2, 3, 4], [2, 3, 4, 1, 1, 1], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

@test L.complete_graph_ijv(1) == L.empty_graph_ijv(1)
@test complete_graph(1) == empty_graph(1)
L.complete_graph_ijv(5)

@test L.complete_bipartite_graph(1) == L.sparse([0 1 ; 1 0])
@test L.complete_bipartite_graph(0) == L.empty_graph(0)
@test L.complete_bipartite_graph_ijv(1) == L.IJV{Float64,Int64}(2, 2, [2, 1], [1, 2], [1.0, 1.0])
@test L.complete_bipartite_graph_ijv(0) == L.empty_graph_ijv(0)

@test L.ring_graph_ijv(1) == L.empty_graph_ijv(1)
@test ring_graph(1) == empty_graph(1)
ring_graph(5)

generalized_ring(11,[1 2 5])
@test generalized_ring(5,[5 10]) == empty_graph(5)
@test generalized_ring(1,[5 10]) == empty_graph(1)

rand_gen_ring(100,3)
rand_gen_ring(100,3,verbose=true)

@test hypercube(1) == path_graph(2)
hypercube(3)

a = L.complete_graph_ijv(5)
b = L.cbt_ijv(4)*3
ab = product_graph(a,b);

@test sparse(ab) == product_graph(sparse(a),sparse(b))

grid2(2,3,isotropy=2.0)
grid2(3)

grid3(3)
grid3(2,3,4)

Random.seed!(1)
rand_matching(4)
rand_matching(5)

Random.seed!(1)
rand_regular(7,1)
rand_regular(7,2)
rand_regular(6,1)
rand_regular(6,3)

Random.seed!(1)
grown_graph(11,1)
grown_graph(12,2)

Random.seed!(1)
grown_graph_d(50,2)
grown_graph_d(10,8)

Random.seed!(1)
pref_attach(6,5,0.5)
randperm(pref_attach(30,2,0.3))

Random.seed!(1)
L.ErdosRenyi_ijv(10,20)

Random.seed!(1)
L.ErdosRenyiCluster_ijv(6,30)
L.ErdosRenyiCluster_ijv(30,10)

Random.seed!(1)
L.ErdosRenyiClusterFix_ijv(20,20)
L.ErdosRenyiClusterFix_ijv(10,50)

ErdosRenyiClusterFix(10,70)

L.firstn(L.grid2_ijv(10),45)


L.randperm_ver(L.V06, path_graph(6))

semiwted_chimera(10,1, ver=L.V06)


# check if still doing what V06 should do

RandomV06.seed_ver!(RandomV06.V06, 1); 
v = randn_ver(Laplacians.V06, 2000);

s = 0; 
for i in 1:1000
    global s
    global a
    a = chimera(2000, i, ver=Laplacians.V06)
    s += v'*a*v;
end

@test isapprox(s, 2784.2498256003364)


s = 0; 
for i in 1:1000
    global s
    global a
    a = wted_chimera(2000, i, ver=Laplacians.V06)
    s += v'*a*v;
end

@test abs(s - 8649.442267154831) < 1e-5

latin_square_graph(5)
S = latin_square(5)
latin_square_graph(S)



;




