using SparseArrays
using Laplacians

mutable struct LLp{Tind,Tval} # the node of a single linked list
  row::Tind
  val::Tval
  next::LLp{Tind,Tval}
  reverse::LLp{Tind,Tval}

  LLp{Tind,Tval}() where {Tind,Tval} = (x = new(zero(Tind), zero(Tval)); x.next = x; x.reverse = x)
  LLp{Tind,Tval}(row, val, next, rev) where {Tind,Tval} = new(row, val, next, rev)
  LLp{Tind,Tval}(row, val) where {Tind,Tval} = (x = new(row, val); x.next = x; x.reverse = x)
  LLp{Tind,Tval}(row, val, next) where {Tind,Tval} = (x = new(row, val, next); x.reverse = x)
end

"""
LLmatp is the data structure used to maintain the matrix during elimination.
It stores the elements in each column in a singly linked list (only next ptrs)
Each element is an LLp (linked list pointer).
The head of each column is pointed to by cols.

We probably can get rid of degs - as it is only used to store initial degrees.
"""
mutable struct LLmatp{Tind,Tval}
  n::Int64
  degs::Array{Tind,1}
  cols::Array{LLp{Tind,Tval},1}
  lles::Array{LLp{Tind,Tval},1} # a serialized reference (when constructed the order is the same)
end

function LLmatp(a::SparseMatrixCSC{Tval,Tind}) where {Tind,Tval}
  n = size(a,1) # n is the # of rows in a; a is the adj matrix
  m = nnz(a) # m is the # of nnz elements in a

  degs = zeros(Tind,n) # each node in A is allocated an entry; Tind are Int64 if the default ctor is used

  flips = flipIndex(a) # which nz values are the symmetric conterpart of x (the idx of nzval vector)

  cols = Array{LLp{Tind,Tval}}(undef, n) # each column is transformed into a linked list
  llelems = Array{LLp{Tind,Tval}}(undef, m)

  for i in 1:n
      degs[i] = a.colptr[i+1] - a.colptr[i] # the # of element for a col in A is the degree of that node

      ind = a.colptr[i] # in each linked list of a column
      j = a.rowval[ind]
      v = a.nzval[ind]
      llpend = LLp{Tind,Tval}(j,v) # setup the first element. llp end
      next = llelems[ind] = llpend
      for ind in (a.colptr[i]+one(Tind)):(a.colptr[i+1]-one(Tind)) # for all elements in that column
          j = a.rowval[ind]
          v = a.nzval[ind]
          next = llelems[ind] = LLp{Tind,Tval}(j,v,next) # (this) -> (next); next = this;
      end
      cols[i] = next # let the cols[i] points to the linked list
  end

  for i in 1:n # for each column
      for ind in a.colptr[i]:(a.colptr[i+1]-one(Tind))
          llelems[ind].reverse = llelems[flips[ind]] # the reverse points to the symmetric point (the other entry) of this edge
      end
  end

  return LLmatp{Tind,Tval}(n, degs, cols, llelems)
end

function flipIndex(a::SparseMatrixCSC{Tval,Tind}) where {Tval,Tind}

  b = SparseMatrixCSC(a.m, a.n, copy(a.colptr), copy(a.rowval), collect(UnitRange{Tind}(1,nnz(a))) );
  bakMat = copy(b');
  return bakMat.nzval

end


s = sparse([0 1.0 0; 1 0 1; 0 1 0])

println(s)
println(s.colptr)
println(s.nzval)

sllmatp = LLmatp(s)

sldli = Laplacians.approxChol(sllmatp)
print(sldli)