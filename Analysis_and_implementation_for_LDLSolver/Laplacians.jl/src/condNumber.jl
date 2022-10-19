#=
This is old, but is being maintained for some backwards compatability.
Generally more useful routines are in conditionNumber.

Compute the condition number given the initial graph and the preconditioner.
Uses cholfact, so, if the initial graph is not sdd or if inverse has a lot of nonzeros this fails. 
=#
function condNumber(la::SparseMatrixCSC{Tv,Ti}, prec::SparseMatrixCSC{Tv,Ti}; tol = 1e-5) where {Tv,Ti}
	# cf = cholfact(prec)
	n = la.n
	ldl = ldlt(prec)
	diag = 1 ./ (ldl.D \ ones(n))

	return computeCN(la, ldl.UP , diag, tol=tol)
end

# M = Ut * d * U
function condNumber(la::SparseMatrixCSC{Tv,Ti}, U::UpperTriangular{Tv,SparseMatrixCSC{Tv,Ti}}, d::Array{Tv,1}; tol = 1e-5) where {Tv,Ti}
	return computeCN(la, U, d, tol=tol)
end

# M = Ut * U
function condNumber(la::SparseMatrixCSC{Tv,Ti}, U::UpperTriangular{Tv,SparseMatrixCSC{Tv,Ti}}; tol = 1e-5) where {Tv,Ti}
	return computeCN(la, U, ones(a.n), tol=tol)
end

function computeCN(la::SparseMatrixCSC{Tv,Ti}, U, d::Array{Tv,1}; tol=1e-5) where {Tv,Ti}

	n = la.n

	# get the max eigenvalue of d^-1/2 * L^-1 * a * U^-1 * d^-1/2
	f = function(b)
		res = copy(b)

		# diag sqrt inverse 
		for i in 1:(n - 1)
            res[i] = res[i] * d[i]^(-1/2)
        end
		res[n] = 0

		res = U \ res
		res = la * res
		res = U' \ res

		# diag sqrt inverse 
        for i in 1:(n - 1)
            res[i] = res[i] * d[i]^(-1/2)
        end
        res[n] = 0

		return res
	end
	fOp = SqLinOp(true,1.0,n,f)
	lmax = eigs(fOp;nev=1,which=:LM,tol=0.0)[1][1]

	if lmax < 1e-20
		# If lambdamax is too small, the computation of 1 - 1/K won't be possible. 
		# So, just return a huge condition number (which should be true anyway)
		return 1,1e20
	end

	# get the max eigenvalue of 1 - (M^-1/2 L M^-1/2) / lambdaMax
	g = function(b)
		res2 = copy(b)
		res = copy(b) / lmax

		# diag sqrt inverse
		for i in 1:(n - 1)
            res[i] = res[i] * d[i]^(-1/2)
        end
		res[n] = 0

		res = U \ res
		res = la * res
		res = U' \ res

		# diag sqrt inverse
		for i in 1:(n - 1)
            res[i] = res[i] * d[i]^(-1/2)
        end
		res[n] = res2[n] = 0

		return res2 - res
	end
	gOp = SqLinOp(true,1.0,n,g)

    #R = eigs(gOp;nev=1,which=:LM,tol=tol)[1][1]
	R = powerIteration(gOp, tol=tol, maxit=100)

    Kmin = 1 / (1 - R)
    Kmax = 1 / (1 - R - tol)

    if Kmax < 0
    	while Kmax < 0
    		tol = tol / 10

    		R = powerIteration(gOp, tol=tol)

		    Kmin = 1 / (1 - R)
		    Kmax = 1 / (1 - R - tol)
    	end

    	println("Initial tolerance too large. Recomputed with tolerance = ", tol)
    end

    return (Kmin, Kmax)

end


# function computeCN{Tv,Ti}(la::SparseMatrixCSC{Tv,Ti}, U::UpperTriangular{Tv,SparseMatrixCSC{Tv,Ti}}, d::Array{Tv,1}; tol=1e-3)

# 	n = la.n

# 	# get the max eigenvalue of d^-1/2 * L^-1 * a * U^-1 * d^-1/2
# 	f = function(b::Array{Float64,1})
# 		res = copy(b)

# 		# diag sqrt inverse 
# 		res = res .* d.^-0.5
# 		res[n] = 0

# 		res = U \ res
# 		res = la * res
# 		res = U' \ res

# 		# diag sqrt inverse 
#         res = res .* d.^-0.5
#         res[n] = 0

# 		return res
# 	end
# 	fOp = SqLinOp(true,1.0,n,f)
# 	lmax = eigs(fOp;nev=1,which=:LM,tol=0.0)[1][1]

# 	# get the max eigenvalue of 1 - (M^-1/2 L M^-1/2) / lambdaMax
# 	g = function(b::Array{Float64,1})
# 		res2 = copy(b)
# 		res = copy(b) / lmax

# 		# diag sqrt inverse
# 		res = res .* d.^-0.5
# 		res[n] = 0

# 		res = U \ res
# 		res = la * res
# 		res = U' \ res

# 		# diag sqrt inverse
# 		res = res .* d.^-0.5
# 		res[n] = res2[n] = 0

# 		return res2 - res
# 	end
# 	gOp = SqLinOp(true,1.0,n,g)

# 	R = eigs(gOp;nev=1,which=:LM,tol=tol)[1][1]

#     Kmin = 1 / (1 - R)
#     Kmax = 1 / (1 - R - tol)

#     return (Kmin, Kmax)

# end
