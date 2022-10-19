#=

approxChol Laplacian solver by Daniel A. Spielman, 2017.
This algorithm is an implementation of an approximate edge-by-edge elimination
algorithm inspired by the Approximate Gaussian Elimination algorithm of
Kyng and Sachdeva.

For usage exaples, see http://danspielman.github.io/Laplacians.jl/latest/usingSolvers/index.html

There are two versions of this solver:
one that fixes the order of elimination beforehand,
and one that adapts the order to eliminate verties of low degree.
These use different data structures.
LLOrdMat is for the fixed order, and LLmatp is for the adaptive order.

These coes produce a structure we call LDLinv that is then used in the solve.
The structure of this code is as follows:

The data structures appear in approxCholTypes.jl
We then have the outline:

* constructors for LLmatp and LLMatOrd
* get_ll_col and compress_ll_col : used inside the elimination
* approxChol : the main routine
* LDLsolver, and its forward and backward solve the apply LDLinv
* approxchol_lap: the main solver, which calls approxchol_lap1 on connected
    components.
    This then calls one of approxchol_lapWdeg, approxchol_lapGiven or approxchol_lapGreedy,
    depending on the parameters.

* approxchol_lapChol - for producing a Cholesky factor instead of an LDLinv.
  might be useful if optimized.
* data structures that are used for the adaptive low-degree version to
  choose the next vertex.

=#

"""
    params = ApproxCholParams(order, output)
order can be one of
* :deg (by degree, adaptive),
* :wdeg (by original wted degree, nonadaptive),
* :given
"""
mutable struct ApproxCholParams
    order::Symbol
    stag_test::Integer
end

ApproxCholParams() = ApproxCholParams(:deg, 5)
ApproxCholParams(sym::Symbol) = ApproxCholParams(sym, 5)

LDLinv(a::SparseMatrixCSC{Tval,Tind}) where {Tind,Tval} =
  LDLinv(zeros(Tind,a.n-1), zeros(Tind,a.n),Tind[],Tval[],zeros(Tval,a.n))

LDLinv(a::LLMatOrd{Tind,Tval}) where {Tind,Tval} =
  LDLinv(zeros(Tind,a.n-1), zeros(Tind,a.n),Tind[],Tval[],zeros(Tval,a.n))

LDLinv(a::LLmatp{Tind,Tval}) where {Tind,Tval} =
  LDLinv(zeros(Tind,a.n-1), zeros(Tind,a.n),Tind[],Tval[],zeros(Tval,a.n))


function LLmatp(a::SparseMatrixCSC{Tval,Tind}) where {Tind,Tval}
    n = size(a,1) # n is the # of rows in a; a is the adj matrix
    m = nnz(a) # m is the # of nnz elements in a

    degs = zeros(Tind,n) # each node in A is allocated an entry; Tind are Int64 if the default ctor is used

    flips = flipIndex(a) # which nz values are the symmetric conterpart of x (the idx of nzval vector)

    cols = Array{LLp{Tind,Tval}}(undef, n) # each column is transformed into a linked list
    llelems = Array{LLp{Tind,Tval}}(undef, m)

    @inbounds for i in 1:n
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

    @inbounds for i in 1:n # for each column
        for ind in a.colptr[i]:(a.colptr[i+1]-one(Tind))
            llelems[ind].reverse = llelems[flips[ind]] # the reverse points to the symmetric point (the other entry) of this edge
        end
    end

    return LLmatp{Tind,Tval}(n, degs, cols, llelems)
end

"""
  Print a column in an LLmatp matrix.
  This is here for diagnostics.
"""
function print_ll_col(llmat::LLmatp, i::Int)
    ll = llmat.cols[i]
    println("col $i, row $(ll.row) : $(ll.val)")

    while ll.next != ll
        ll = ll.next
        println("col $i, row $(ll.row) : $(ll.val)")
    end
end

function LLMatOrd(a::SparseMatrixCSC{Tval,Tind}) where {Tind,Tval}
    n = size(a,1)
    m = nnz(a)

    cols = zeros(Tind, n)
    llelems = Array{LLord{Tind,Tval}}(undef, m)

    ptr = one(Tind)

    @inbounds for i in Tind(1):Tind(n-1)
        next = zero(Tind)

        for ind in (a.colptr[i]):(a.colptr[i+1]-one(Tind))
            j = a.rowval[ind]
            if (i < j)

              v = a.nzval[ind]
              llelems[ptr] = LLord{Tind,Tval}(j, next, v)
              next = ptr
              ptr += one(Tind)

            end
        end
        cols[i] = next
    end

    return LLMatOrd{Tind,Tval}(n, cols, llelems)
end

function LLMatOrd(a::SparseMatrixCSC{Tval,Tind}, perm::Array) where {Tind,Tval}
    n = size(a,1)
    m = nnz(a)

    invp = invperm(perm)

    cols = zeros(Tind, n)
    llelems = Array{LLord{Tind,Tval}}(undef, m)

    ptr = one(Tind)

    @inbounds for i0 in Tind(1):Tind(n)
        i = invp[i0]
        next = zero(Tind)

        for ind in (a.colptr[i0]):(a.colptr[i0+1]-one(Tind))
            j = invp[a.rowval[ind]]
            if (i < j)

              v = a.nzval[ind]
              llelems[ptr] = LLord{Tind,Tval}(j, next, v)
              next = ptr
              ptr += one(ptr)

            end
        end
        cols[i] = next
    end

    return LLMatOrd{Tind,Tval}(n, cols, llelems)
end

"""
  Print a column in an LLMatOrd matrix.
  This is here for diagnostics.
"""
function print_ll_col(llmat::LLMatOrd, i::Int)
    ptr = llmat.cols[i]
    while ptr != 0
      ll = llmat.lles[ptr]
      println("col $i, row $(ll.row) : $(ll.val)")

      ptr = ll.next
    end
end



#=============================================================

The approximate factorization

=============================================================#

function get_ll_col(llmat::LLmatp{Tind,Tval},
  i,
  colspace::Vector{LLp{Tind,Tval}}) where {Tind,Tval}
  # compactly put the references for all elements in a column to a vector

    ll = llmat.cols[i] # the ith column
    len = 0
    @inbounds while ll.next != ll

        if ll.val > zero(Tval)
            len = len+1
            if (len > length(colspace))
                push!(colspace,ll)
            else
                colspace[len] = ll
            end
        end

        ll = ll.next
    end

    if ll.val > zero(Tval) # store the last element
        len = len+1
        if (len > length(colspace))
            push!(colspace,ll)
        else
            colspace[len] = ll
        end
    end

    return len # return the number of nonzero elements
end

function get_ll_col(llmat::LLMatOrd{Tind,Tval},
  i,
  colspace::Vector{LLcol{Tind,Tval}}) where {Tind,Tval}

    ptr = llmat.cols[i]
    len = 0
    @inbounds while ptr != 0

        #if ll.val > 0
            len = len+1

            # should not be an lles - is an abuse
            item = LLcol(llmat.lles[ptr].row, ptr, llmat.lles[ptr].val)
            if (len > length(colspace))
                push!(colspace,item)
            else
                colspace[len] = item
            end
        #end

        ptr = llmat.lles[ptr].next
    end

    return len
end

function compressCol!(a::LLmatp{Tind,Tval},
  colspace::Vector{LLp{Tind,Tval}},
  len::Int
  ) where {Tind,Tval}

    o = Base.Order.ord(isless, x->x.row, false, Base.Order.Forward)

    sort!(colspace, 1, len, QuickSort, o) # sort the colspace based on the row value (ascending order)

    ptr = 0
    currow::Tind = 0

    c = colspace

    @inbounds for i in 1:len

        if c[i].row != currow # if elements are not in the same row
            currow = c[i].row
            ptr = ptr+1
            c[ptr] = c[i]

        else
            c[ptr].val = c[ptr].val + c[i].val
            c[i].reverse.val = zero(Tval) # basically delete, because it will not be retrived by get_ll_col method

            # approxCholPQDec!(pq, currow)
        end
    end


    o = Base.Order.ord(isless, x->x.val, false, Base.Order.Forward) # sort based on nodes' value (the degree will always be the last)
    sort!(colspace, 1, ptr, QuickSort, o) # ascending order

    return ptr
end


function compressCol!(a::LLmatp{Tind,Tval},
  colspace::Vector{LLp{Tind,Tval}},
  len::Int,
  pq::ApproxCholPQ{Tind}) where {Tind,Tval}

    o = Base.Order.ord(isless, x->x.row, false, Base.Order.Forward)

    sort!(colspace, 1, len, QuickSort, o) # sort the colspace based on the row value (ascending order)

    ptr = 0
    currow::Tind = 0

    c = colspace

    @inbounds for i in 1:len

        if c[i].row != currow # if elements are not in the same row
            currow = c[i].row
            ptr = ptr+1
            c[ptr] = c[i]

        else
            c[ptr].val = c[ptr].val + c[i].val
            c[i].reverse.val = zero(Tval) # basically delete, because it will not be retrived by get_ll_col method

            approxCholPQDec!(pq, currow)
        end
    end


    o = Base.Order.ord(isless, x->x.val, false, Base.Order.Forward) # sort based on nodes' value (the degree will always be the last)
    sort!(colspace, 1, ptr, QuickSort, o) # ascending order

    return ptr
end

function compressCol!(
  colspace::Vector{LLcol{Tind,Tval}},
  len::Int
  ) where {Tind,Tval}

    o = Base.Order.ord(isless, x->x.row, false, Base.Order.Forward)

    sort!(colspace, one(len), len, QuickSort, o)

    c = colspace

    ptr = 0
    currow = c[1].row
    curval = c[1].val
    curptr = c[1].ptr

    @inbounds for i in 2:len

        if c[i].row != currow

            ptr = ptr+1
            c[ptr] = LLcol(currow, curptr, curval)  # next is abuse here: reall keep where it came from.

            currow = c[i].row
            curval = c[i].val
            curptr = c[i].ptr

        else

            curval = curval + c[i].val

        end

    end

    # emit the last row

    ptr = ptr+1
    c[ptr] = LLcol(currow, curptr, curval)

    o = Base.Order.ord(isless, x->x.val, false, Base.Order.Forward)
    # sort!(colspace, one(ptr), ptr, QuickSort, o)
    sort!(colspace, one(ptr), ptr, MergeSort, o) # use a stable sort

    return ptr
end


function approxChol(a::LLMatOrd{Tind,Tval}) where {Tind,Tval}
    n = a.n

    # need to make custom one without col info later.
    ldli = LDLinv(a)
    ldli_row_ptr = one(Tind)

    d = zeros(Tval,n)

    colspace = Array{LLcol{Tind,Tval}}(undef, n)
    cumspace = Array{Tval}(undef, n)
    #vals = Array(Tval,n) # will be able to delete this

    o = Base.Order.ord(isless, identity, false, Base.Order.Forward)


    for i in Tind(1):Tind(n-1)

        ldli.col[i] = i  # will get rid of this with new data type
        ldli.colptr[i] = ldli_row_ptr

        len = get_ll_col(a, i, colspace)

        len = compressCol!(colspace, len)

        csum = zero(Tval)
        for ii in 1:len
            #vals[ii] = colspace[ii].val    # if immut, no need for vals
            csum = csum + colspace[ii].val
            cumspace[ii] = csum
        end
        wdeg = csum

        colScale = one(Tval)

        for joffset in 1:(len-1)

            llcol = colspace[joffset]
            w = llcol.val * colScale
            j = llcol.row

            f = w/(wdeg)

            #vals[joffset] = zero(Tval)

            # kind = Laplacians.blockSample(vals,k=1)[1]
            r = rand() * (csum - cumspace[joffset]) + cumspace[joffset]
            koff = searchsortedfirst(cumspace,r,one(len),len,o)

            k = colspace[koff].row

            newEdgeVal = w*(one(Tval)-f)

            # create edge (j,k) with newEdgeVal
            # do it by reassigning ll
            if j < k # put it in col j
                jhead = a.cols[j]
                a.lles[llcol.ptr] = LLord(k, jhead, newEdgeVal)
                #ll.next = jhead
                #ll.val = newEdgeVal
                #ll.row = k
                a.cols[j] = llcol.ptr
            else # put it in col k
              khead = a.cols[k]
              a.lles[llcol.ptr] = LLord(j, khead, newEdgeVal)
              #ll.next = khead
              #ll.val = newEdgeVal
              #ll.row = j
              a.cols[k] = llcol.ptr
            end

            colScale = colScale*(one(Tval)-f)
            #wdeg = wdeg*(1.0-f)^2
            wdeg = wdeg - 2w + w^2/wdeg

            push!(ldli.rowval,j)
            push!(ldli.fval, f)
            ldli_row_ptr = ldli_row_ptr + one(Tind)

            # push!(ops, IJop(i,j,1-f,f))  # another time suck


        end # for joffset


        llcol = colspace[len]
        w = llcol.val * colScale
        j = llcol.row

        push!(ldli.rowval,j)
        push!(ldli.fval, one(Tval))
        ldli_row_ptr = ldli_row_ptr + one(Tind)

        d[i] = w

    end # for i


    ldli.colptr[n] = ldli_row_ptr

    ldli.d = d

    return ldli
end

# this one is greedy on the degree - also a big win
function approxChol_reordered(a::LLmatp{Tind,Tval}, order::Vector{Tind}) where {Tind,Tval}
    n = a.n

    ldli = LDLinv(a)
    ldli_row_ptr = one(Tind) # keep track of current starting point of a row

    d = zeros(n)

    # pq = ApproxCholPQ(a.degs) # this gives the order to do guassian elimination

    it = 1

    colspace = Array{LLp{Tind,Tval}}(undef, n)
    cumspace = Array{Tval}(undef, n)
    vals = Array{Tval}(undef, n) # will be able to delete this

    o = Base.Order.ord(isless, identity, false, Base.Order.Forward)

    @inbounds while it < n


        # i = approxCholPQPop!(pq) # i is column as well as node
        i = order[it]

        # the <it> column of the ldli is set to node <i>
        ldli.col[it] = i # conversion! // i is not it, i is the column that is eliminated during it
        ldli.colptr[it] = ldli_row_ptr # log the start point of the column

        it = it + 1

        len = get_ll_col(a, i, colspace)

        len = compressCol!(a, colspace, len) #, pq)  #3hog

        csum = zero(Tval) # the sum of all the weight? [should be, the sampling prob is w/wdeg]
        for ii in 1:len
            vals[ii] = colspace[ii].val
            csum = csum + colspace[ii].val
            cumspace[ii] = csum
        end
        wdeg = csum

        colScale = one(Tval)

        for joffset in 1:(len-1) # for each edge ??

            ll = colspace[joffset]
            w = vals[joffset] * colScale
            j = ll.row
            revj = ll.reverse # the column j

            f = w/(wdeg) # guassian elemiantion's factor

            vals[joffset] = zero(Tval)

            # kind = Laplacians.blockSample(vals,k=1)[1]
            # sum is a linear space [xxxxxxxx|bbbb|aa], rand() * (csum - cumspace[joffset]) locate a point in the
            # segment
            r = rand() * (csum - cumspace[joffset]) + cumspace[joffset] # r is a value >= cumspace[joffset]
            koff = searchsortedfirst(cumspace,r,one(len),len,o) # find the first ele that is greater than r
            # koff is the random sampling result

            k = colspace[koff].row # k is the row, also the other node idx for that sampled edge

            # approxCholPQInc!(pq, k) # that node get a new edge

            newEdgeVal = f*(one(Tval)-f)*wdeg # w * remaining w / total w

            # the fix is done by changing && moving the nodes to corresponding columns
            # if there is an edge, this extra node will be merged during compressCol! routine
            # but where is the degree??

            # fix row k in col j
            revj.row = k   # dense time hog: presumably becaus of cache
            revj.val = newEdgeVal
            revj.reverse = ll

            # fix row j in col k
            khead = a.cols[k]
            a.cols[k] = ll
            ll.next = khead
            ll.reverse = revj
            ll.val = newEdgeVal
            ll.row = j

            colScale = colScale*(one(Tval)-f) # ?? why this? 
            wdeg = wdeg*(one(Tval)-f)^2 # wremain * colScale

            # the operation is logged into ldli datastructure.

            push!(ldli.rowval,j) # we can see that ldli_row_ptr && colptr is associated with rowval & fval vector
            push!(ldli.fval, f)
            ldli_row_ptr = ldli_row_ptr + one(Tind)

            # push!(ops, IJop(i,j,1-f,f))  # another time suck


        end # for


        if len > 0
          ll = colspace[len]
          w = vals[len] * colScale
          j = ll.row
          revj = ll.reverse

          # if it < n
          #     approxCholPQDec!(pq, j)
          # end

          revj.val = zero(Tval)

          push!(ldli.rowval,j)
          push!(ldli.fval, one(Tval))
          ldli_row_ptr = ldli_row_ptr + one(Tind)

          d[i] = w
        else # default to 1
          d[i] = 1
        end

    end

    ldli.colptr[it] = ldli_row_ptr

    ldli.d = d

    return ldli
end

# this one is greedy on the degree - also a big win
function approxChol_reordered(a::LLmatp{Tind,Tval}, order::Vector{Tind}, rnums::Vector{Float64}) where {Tind,Tval}
    n = a.n

    ldli = LDLinv(a)
    ldli_row_ptr = one(Tind) # keep track of current starting point of a row

    d = zeros(n)

    # pq = ApproxCholPQ(a.degs) # this gives the order to do guassian elimination

    it = 1

    colspace = Array{LLp{Tind,Tval}}(undef, n)
    cumspace = Array{Tval}(undef, n)
    vals = Array{Tval}(undef, n) # will be able to delete this

    o = Base.Order.ord(isless, identity, false, Base.Order.Forward)

    curR = 1

    @inbounds while it < n

        # i = approxCholPQPop!(pq) # i is column as well as node
        i = order[it]

        # the <it> column of the ldli is set to node <i>
        ldli.col[it] = i # conversion! // i is not it, i is the column that is eliminated during it
        ldli.colptr[it] = ldli_row_ptr # log the start point of the column

        it = it + 1

        len = get_ll_col(a, i, colspace)

        len = compressCol!(a, colspace, len) #, pq)  #3hog

        csum = zero(Tval) # the sum of all the weight? [should be, the sampling prob is w/wdeg]
        # print("$(i): ")
        for ii in 1:len
          # print("[$(colspace[ii].val) $(colspace[ii].row)] ")
            vals[ii] = colspace[ii].val
            csum = csum + colspace[ii].val
            cumspace[ii] = csum
        end
        # println("")
        wdeg = csum

        colScale = one(Tval)

        for joffset in 1:(len-1) # for each edge ??

            ll = colspace[joffset]
            w = vals[joffset] * colScale
            j = ll.row
            revj = ll.reverse # the column j

            f = w/(wdeg) # guassian elemiantion's factor

            vals[joffset] = zero(Tval)

            # kind = Laplacians.blockSample(vals,k=1)[1]
            # sum is a linear space [xxxxxxxx|bbbb|aa], rand() * (csum - cumspace[joffset]) locate a point in the
            # segment
            rn = rnums[curR] # get a random number
            curR += 1

            r = rn * (csum - cumspace[joffset]) + cumspace[joffset] # r is a value >= cumspace[joffset]
            koff = searchsortedfirst(cumspace,r,one(len),len,o) # find the first ele that is greater than r
            # koff is the random sampling result

            k = colspace[koff].row # k is the row, also the other node idx for that sampled edge

            # approxCholPQInc!(pq, k) # that node get a new edge

            newEdgeVal = f*(one(Tval)-f)*wdeg # w * remaining w / total w

            # the fix is done by changing && moving the nodes to corresponding columns
            # if there is an edge, this extra node will be merged during compressCol! routine
            # but where is the degree??

            # fix row k in col j
            # println("($(j),$(k)): $(newEdgeVal)")
            revj.row = k   # dense time hog: presumably becaus of cache
            revj.val = newEdgeVal
            revj.reverse = ll

            # fix row j in col k
            khead = a.cols[k]
            a.cols[k] = ll
            ll.next = khead
            ll.reverse = revj
            ll.val = newEdgeVal
            ll.row = j

            colScale = colScale*(one(Tval)-f) # ?? why this? 
            wdeg = wdeg*(one(Tval)-f)^2 # wremain * colScale

            # the operation is logged into ldli datastructure.

            push!(ldli.rowval,j) # we can see that ldli_row_ptr && colptr is associated with rowval & fval vector
            push!(ldli.fval, f)
            ldli_row_ptr = ldli_row_ptr + one(Tind)

            # push!(ops, IJop(i,j,1-f,f))  # another time suck


        end # for


        if len > 0
          ll = colspace[len]
          w = vals[len] * colScale
          j = ll.row
          revj = ll.reverse

          # if it < n
          #     approxCholPQDec!(pq, j)
          # end

          revj.val = zero(Tval)

          push!(ldli.rowval,j)
          push!(ldli.fval, one(Tval))
          ldli_row_ptr = ldli_row_ptr + one(Tind)

          d[i] = w
        else # default to 1
          d[i] = 1
        end

    end

    ldli.colptr[it] = ldli_row_ptr

    ldli.d = d

    return ldli
end

# this one is greedy on the degree - also a big win
function approxChol_log(a::LLmatp{Tind,Tval}) where {Tind,Tval}
    n = a.n

    ldli = LDLinv(a)
    ldli_row_ptr = one(Tind) # keep track of current starting point of a row

    d = zeros(n)

    pq = ApproxCholPQ(a.degs) # this gives the order to do guassian elimination

    it = 1

    colspace = Array{LLp{Tind,Tval}}(undef, n)
    cumspace = Array{Tval}(undef, n)
    vals = Array{Tval}(undef, n) # will be able to delete this
    log = Vector{Int}(undef, n - 1)

    o = Base.Order.ord(isless, identity, false, Base.Order.Forward)

    @inbounds while it < n
        log[it] = pq.deg1 # track the deg1

        i = approxCholPQPop!(pq) # i is column as well as node

        # the <it> column of the ldli is set to node <i>
        ldli.col[it] = i # conversion! // i is not it, i is the column that is eliminated during it
        ldli.colptr[it] = ldli_row_ptr # log the start point of the column

        it = it + 1

        len = get_ll_col(a, i, colspace)

        len = compressCol!(a, colspace, len, pq)  #3hog

        csum = zero(Tval) # the sum of all the weight? [should be, the sampling prob is w/wdeg]
        for ii in 1:len
            vals[ii] = colspace[ii].val
            csum = csum + colspace[ii].val
            cumspace[ii] = csum
        end
        wdeg = csum

        colScale = one(Tval)

        for joffset in 1:(len-1) # for each edge ??

            ll = colspace[joffset]
            w = vals[joffset] * colScale
            j = ll.row
            revj = ll.reverse # the column j

            f = w/(wdeg) # guassian elemiantion's factor

            vals[joffset] = zero(Tval)

            # kind = Laplacians.blockSample(vals,k=1)[1]
            # sum is a linear space [xxxxxxxx|bbbb|aa], rand() * (csum - cumspace[joffset]) locate a point in the
            # segment
            r = rand() * (csum - cumspace[joffset]) + cumspace[joffset] # r is a value >= cumspace[joffset]
            koff = searchsortedfirst(cumspace,r,one(len),len,o) # find the first ele that is greater than r
            # koff is the random sampling result

            k = colspace[koff].row # k is the row, also the other node idx for that sampled edge

            approxCholPQInc!(pq, k) # that node get a new edge

            newEdgeVal = f*(one(Tval)-f)*wdeg # w * remaining w / total w

            # the fix is done by changing && moving the nodes to corresponding columns
            # if there is an edge, this extra node will be merged during compressCol! routine
            # but where is the degree??

            # fix row k in col j
            revj.row = k   # dense time hog: presumably becaus of cache
            revj.val = newEdgeVal
            revj.reverse = ll

            # fix row j in col k
            khead = a.cols[k]
            a.cols[k] = ll
            ll.next = khead
            ll.reverse = revj
            ll.val = newEdgeVal
            ll.row = j

            colScale = colScale*(one(Tval)-f) # ?? why this? 
            wdeg = wdeg*(one(Tval)-f)^2 # wremain * colScale

            # the operation is logged into ldli datastructure.

            push!(ldli.rowval,j) # we can see that ldli_row_ptr && colptr is associated with rowval & fval vector
            push!(ldli.fval, f)
            ldli_row_ptr = ldli_row_ptr + one(Tind)

            # push!(ops, IJop(i,j,1-f,f))  # another time suck


        end # for


        ll = colspace[len]
        w = vals[len] * colScale
        j = ll.row
        revj = ll.reverse

        if it < n
            approxCholPQDec!(pq, j)
        end

        revj.val = zero(Tval)

        push!(ldli.rowval,j)
        push!(ldli.fval, one(Tval))
        ldli_row_ptr = ldli_row_ptr + one(Tind)

        d[i] = w

    end

    ldli.colptr[it] = ldli_row_ptr

    ldli.d = d

    return ldli, log
end

# this one is greedy on the degree - also a big win
function approxChol(a::LLmatp{Tind,Tval}) where {Tind,Tval}
    n = a.n

    ldli = LDLinv(a)
    ldli_row_ptr = one(Tind) # keep track of current starting point of a row

    d = zeros(n)

    pq = ApproxCholPQ(a.degs) # this gives the order to do guassian elimination

    it = 1

    colspace = Array{LLp{Tind,Tval}}(undef, n)
    cumspace = Array{Tval}(undef, n)
    vals = Array{Tval}(undef, n) # will be able to delete this

    o = Base.Order.ord(isless, identity, false, Base.Order.Forward)

    @inbounds while it < n

        i = approxCholPQPop!(pq) # i is column as well as node

        # the <it> column of the ldli is set to node <i>
        ldli.col[it] = i # conversion! // i is not it, i is the column that is eliminated during it
        ldli.colptr[it] = ldli_row_ptr # log the start point of the column

        it = it + 1

        len = get_ll_col(a, i, colspace)

        len = compressCol!(a, colspace, len, pq)  #3hog

        csum = zero(Tval) # the sum of all the weight? [should be, the sampling prob is w/wdeg]
        for ii in 1:len
            vals[ii] = colspace[ii].val
            csum = csum + colspace[ii].val
            cumspace[ii] = csum
        end
        wdeg = csum

        colScale = one(Tval)

        for joffset in 1:(len-1) # for each edge ??

            ll = colspace[joffset]
            w = vals[joffset] * colScale
            j = ll.row
            revj = ll.reverse # the column j

            f = w/(wdeg) # guassian elemiantion's factor

            vals[joffset] = zero(Tval)

            # kind = Laplacians.blockSample(vals,k=1)[1]
            # sum is a linear space [xxxxxxxx|bbbb|aa], rand() * (csum - cumspace[joffset]) locate a point in the
            # segment
            r = rand() * (csum - cumspace[joffset]) + cumspace[joffset] # r is a value >= cumspace[joffset]
            koff = searchsortedfirst(cumspace,r,one(len),len,o) # find the first ele that is greater than r
            # koff is the random sampling result

            k = colspace[koff].row # k is the row, also the other node idx for that sampled edge

            approxCholPQInc!(pq, k) # that node get a new edge

            newEdgeVal = f*(one(Tval)-f)*wdeg # w * remaining w / total w

            # the fix is done by changing && moving the nodes to corresponding columns
            # if there is an edge, this extra node will be merged during compressCol! routine
            # but where is the degree??

            # fix row k in col j
            revj.row = k   # dense time hog: presumably becaus of cache
            revj.val = newEdgeVal
            revj.reverse = ll

            # fix row j in col k
            khead = a.cols[k]
            a.cols[k] = ll
            ll.next = khead
            ll.reverse = revj
            ll.val = newEdgeVal
            ll.row = j

            colScale = colScale*(one(Tval)-f) # ?? why this? 
            wdeg = wdeg*(one(Tval)-f)^2 # wremain * colScale

            # the operation is logged into ldli datastructure.

            push!(ldli.rowval,j) # we can see that ldli_row_ptr && colptr is associated with rowval & fval vector
            push!(ldli.fval, f)
            ldli_row_ptr = ldli_row_ptr + one(Tind)

            # push!(ops, IJop(i,j,1-f,f))  # another time suck


        end # for


        ll = colspace[len]
        w = vals[len] * colScale
        j = ll.row
        revj = ll.reverse

        if it < n
            approxCholPQDec!(pq, j)
        end

        revj.val = zero(Tval)

        push!(ldli.rowval,j)
        push!(ldli.fval, one(Tval))
        ldli_row_ptr = ldli_row_ptr + one(Tind)

        d[i] = w

    end

    ldli.colptr[it] = ldli_row_ptr

    ldli.d = d

    return ldli
end

# this one is greedy on the degree - also a big win
function approxChol_alt(a::LLmatp{Tind,Tval}) where {Tind,Tval}
    n = a.n

    ldli = LDLinv(a)
    ldli_row_ptr = one(Tind) # keep track of current starting point of a row

    d = zeros(n)

    pq = ApproxCholPQ(a.degs) # this gives the order to do guassian elimination

    it = 1

    colspace = Array{LLp{Tind,Tval}}(undef, n)
    cumspace = Array{Tval}(undef, n)
    vals = Array{Tval}(undef, n) # will be able to delete this

    o = Base.Order.ord(isless, identity, false, Base.Order.Forward)

    @inbounds while it < n

        i = approxCholPQPop!(pq) # i is column as well as node

        # the <it> column of the ldli is set to node <i>
        ldli.col[it] = i # conversion! // i is not it, i is the column that is eliminated during it
        ldli.colptr[it] = ldli_row_ptr # log the start point of the column

        it = it + 1

        len = get_ll_col(a, i, colspace)

        len = compressCol!(a, colspace, len, pq)  #3hog

        csum = zero(Tval) # the sum of all the weight? [should be, the sampling prob is w/wdeg]
        for ii in 1:len
            vals[ii] = colspace[ii].val
            csum = csum + colspace[ii].val
            cumspace[ii] = csum
        end
        wdeg = csum

        colScale = one(Tval)

        for joffset in 1:(len-1) # for each edge ??

            ll = colspace[joffset]
            w = vals[joffset] * colScale
            j = ll.row
            revj = ll.reverse # the column j

            f = w/(wdeg * colScale) # guassian elemiantion's factor

            vals[joffset] = zero(Tval)

            # kind = Laplacians.blockSample(vals,k=1)[1]
            # sum is a linear space [xxxxxxxx|bbbb|aa], rand() * (csum - cumspace[joffset]) locate a point in the
            # segment
            r = rand() * (csum - cumspace[joffset]) + cumspace[joffset] # r is a value >= cumspace[joffset]
            koff = searchsortedfirst(cumspace,r,one(len),len,o) # find the first ele that is greater than r
            # koff is the random sampling result

            k = colspace[koff].row # k is the row, also the other node idx for that sampled edge

            approxCholPQInc!(pq, k) # that node get a new edge

            newEdgeVal = f*(one(Tval)-f)*wdeg *colScale # w * remaining w / total w

            # the fix is done by changing && moving the nodes to corresponding columns
            # if there is an edge, this extra node will be merged during compressCol! routine
            # but where is the degree??

            # fix row k in col j
            revj.row = k   # dense time hog: presumably becaus of cache
            revj.val = newEdgeVal
            revj.reverse = ll

            # fix row j in col k
            khead = a.cols[k]
            a.cols[k] = ll
            ll.next = khead
            ll.reverse = revj
            ll.val = newEdgeVal
            ll.row = j


            colScale = colScale*(one(Tval)-f) # ?? why this? 
            wdeg = wdeg*(one(Tval)-f)

            # the operation is logged into ldli datastructure.

            push!(ldli.rowval,j) # we can see that ldli_row_ptr && colptr is associated with rowval & fval vector
            push!(ldli.fval, f)
            ldli_row_ptr = ldli_row_ptr + one(Tind)


        end # for


        ll = colspace[len]
        w = vals[len] * colScale
        j = ll.row
        revj = ll.reverse

        if it < n
            approxCholPQDec!(pq, j)
        end

        revj.val = zero(Tval)

        push!(ldli.rowval,j)
        push!(ldli.fval, one(Tval))
        ldli_row_ptr = ldli_row_ptr + one(Tind)

        d[i] = w

    end

    ldli.colptr[it] = ldli_row_ptr

    ldli.d = d

    return ldli
end



#=============================================================

The routines that do the solve.

=============================================================#

function inclusiveScanProd!(v::Vector)
  @inbounds for i in 2:length(v)
    v[i] = v[i - 1] * v[i]
  end
end

function preprocessLDL(ldli::LDLinv{Tind, Tval}) where {Tind, Tval}
    @inbounds for i in 1:(length(ldli.d))
      if ldli.d[i] == 0
        ldli.d[i] = 1
      end
    end

    scales = similar(ldli.fval)
    vecLen = similar(ldli.col)
    o = one(Tind)
    @inbounds for ii in 1:length(ldli.col)
      i = ldli.col[ii] # for every elems in ldli.col
      # the loop processing y from 1 to the second last row
      j0 = ldli.colptr[ii]
      j1 = ldli.colptr[ii+1]-o

      scales[j0] = one(Tval)
      cs = scales[j0]
      for jj in (j0+o):j1
        cs = cs * (one(Tval) - ldli.fval[jj-o])
        scales[jj] = cs
      end

      # do some logging
      vecLen[ii] = j1 - j0 + o
    end

    ldli.fval .*= scales

    return vecLen
end

function LDLsolver(ldli::LDLinv{Tind, Tval}, b::Vector) where {Tind, Tval}
    y = copy(b)

    forward!(ldli, y)

    y ./= ldli.d

    backward!(ldli, y)

    y .-= mean(y)

    return y
end

function forward!(ldli::LDLinv{Tind,Tval}, y::Vector) where {Tind,Tval}

    o = one(Tind)
    @inbounds for ii in 1:length(ldli.col) # length(ldli.col) == a.n - 1
        i = ldli.col[ii] # for every elems in ldli.col
        # the loop processing y from 1 to the second last row

        j0 = ldli.colptr[ii]
        j1 = ldli.colptr[ii+1]-one(Tind)

        yi = y[i] # the element of y in that row

        ############################
        toStore = ldli.fval[j0:j1] .* yi # vectorized load & multiplication

        for i in 1:(j1 - j0 + 1) # gather then scatter
          y[ldli.rowval[j0 + i - 1]] += toStore[i]
        end

        y[i] = yi * ldli.fval[j1] # the loop end to avoid an extra multiplication
    end
end

function backward!(ldli::LDLinv{Tind,Tval}, y::Vector) where {Tind,Tval}
    o = one(Tind)
    @inbounds for ii in length(ldli.col):-1:1
        i = ldli.col[ii]

        j0 = ldli.colptr[ii]
        j1 = ldli.colptr[ii+1]-o

        yi = y[i]

        ############################
        ys = [y[j] for j in ldli.rowval[j0:j1]] # a gather

        partials = ldli.fval[j0:j1] .* ys # vectorized multiplication

        integral = sum(partials) # reduction
        integral += yi * ldli.fval[j1]

        y[i] = integral
    end
end

#=
  An attempt at an efficient solver for the case when y is a matrix.
  Have not yet found a meaningful speedup

function LDLsolver(ldli::LDLinv, b::Matrix)
    y = copy(b)

    (d, n) = size(y)
    @assert n == length(ldli.col)+1

    forward!(ldli, y)

    @inbounds for i in 1:(length(ldli.d))
        if ldli.d[i] != 0
          @simd for j in 1:d
            y[j,i] = y[j,i] / ldli.d[i]
          end
        end
    end

    backward!(ldli, y)

    @inbounds for j in 1:size(y,1)
        mu = mean(y[j,:])

        for i in 1:size(y,2)
            y[j,i] = y[j,i] - mu
        end
    end

    return y
end



function forward!{Tind,Tval}(ldli::LDLinv{Tind,Tval}, y::Matrix)

    (d, n) = size(y)
    @assert n == length(ldli.col)+1

    #yi = zeros(y[:,1])

    @inbounds for ii in 1:length(ldli.col)
        i = ldli.col[ii]

        j0 = ldli.colptr[ii]
        j1 = ldli.colptr[ii+1]-one(Tind)

        for jj in j0:(j1-1)
            j = ldli.rowval[jj]
            @simd for k in 1:d
              y[k,j] = y[k,j] + (ldli.fval[jj] * y[k,i])
              y[k,i] = y[k,i] * (one(Tval)-ldli.fval[jj])
          end
        end
        j = ldli.rowval[j1]

        @simd for k in 1:d
          y[k,j] = y[k,j] + y[k,i]
        end
    end
end

function backward!{Tind,Tval}(ldli::LDLinv{Tind,Tval}, y::Matrix)
    o = one(Tind)

    (d, n) = size(y)
    @assert n == length(ldli.col)+1

    yi = zeros(y[:,1])

    @inbounds for ii in length(ldli.col):-1:1
        i = ldli.col[ii]

        j0 = ldli.colptr[ii]
        j1 = ldli.colptr[ii+1]-o

        j = ldli.rowval[j1]
        #copy!(yi, y[:,i])

        @simd for k in 1:d
          y[k,i] = y[k,i] + y[k,j]
        end

        for jj in (j1-o):-o:j0
            j = ldli.rowval[jj]
            @simd for k in 1:d
              y[k,i] = (one(Tval)-ldli.fval[jj])*y[k,i] + ldli.fval[jj].*y[k,j]
            end
        end
        #y[:,i] = yi
    end
end

=#


"""
    solver = approxchol_lap(a); x = solver(b);
    solver = approxchol_lap(a; tol::Real=1e-6, maxits=1000, maxtime=Inf, verbose=false, pcgIts=Int[], params=ApproxCholParams())

A heuristic by Daniel Spielman inspired by the linear system solver in https://arxiv.org/abs/1605.02353 by Rasmus Kyng and Sushant Sachdeva.  Whereas that paper eliminates vertices one at a time, this eliminates edges one at a time.  It is probably possible to analyze it.
The `ApproxCholParams` let you choose one of three orderings to perform the elimination.

* ApproxCholParams(:given) - in the order given.
    This is the fastest for construction the preconditioner, but the slowest solve.
* ApproxCholParams(:deg) - always eliminate the node of lowest degree.
    This is the slowest build, but the fastest solve.
* ApproxCholParams(:wdeg) - go by a perturbed order of wted degree.

For more info, see http://danspielman.github.io/Laplacians.jl/latest/usingSolvers/index.html
"""
function approxchol_lap(a::SparseMatrixCSC{Tv,Ti};
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams()) where {Tv,Ti}

  if minimum(a.nzval) < 0
      error("Adjacency matrix can not have negative edge weights")
  end

    return Laplacians.lapWrapComponents(approxchol_lap1, a,
    verbose=verbose,
    tol=tol,
    maxits=maxits,
    maxtime=maxtime,
    pcgIts=pcgIts,
    params=params)


end

function approxchol_lapGreedy(a::SparseMatrixCSC;
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams())

  tol_ =tol
  maxits_ =maxits
  maxtime_ =maxtime
  verbose_ =verbose
  pcgIts_ =pcgIts

  t1 = time()

  la = lap(a) # a hit !?

  llmat = LLmatp(a)
  ldli = approxChol(llmat)
  F(b) = LDLsolver(ldli, b)

  if verbose
    println("Using greedy degree ordering. Factorization time: ", time()-t1)
    println("Ratio of operator edges to original edges: ", 2 * length(ldli.fval) / nnz(a))
  end

  if verbose
      println("ratio of max to min diagonal of laplacian : ", maximum(diag(la))/minimum(diag(la)))
  end


  f(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) = pcg(la, b .- mean(b), F, tol=tol, maxits=maxits, maxtime=maxtime, pcgIts=pcgIts, verbose=verbose, stag_test = params.stag_test)

end

function approxchol_lapGiven(a::SparseMatrixCSC;
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams())

  tol_ =tol
  maxits_ =maxits
  maxtime_ =maxtime
  verbose_ =verbose
  pcgIts_ =pcgIts

  t1 = time()

  la = lap(a)

  llmat = LLMatOrd(a)
  ldli = approxChol(llmat)
  F(b) = LDLsolver(ldli, b)

  if verbose
    println("Using given ordering. Factorization time: ", time()-t1)
    println("Ratio of operator edges to original edges: ", 2 * length(ldli.fval) / nnz(a))
  end

  if verbose
      println("ratio of max to min diagonal of laplacian : ", maximum(diag(la))/minimum(diag(la)))
  end


  f(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) = pcg(la, b .- mean(b), F, tol=tol, maxits=maxits, maxtime=maxtime, pcgIts=pcgIts, verbose=verbose, stag_test = params.stag_test)

end

function approxchol_lapWdeg(a::SparseMatrixCSC;
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams())

  tol_ =tol
  maxits_ =maxits
  maxtime_ =maxtime
  verbose_ =verbose
  pcgIts_ =pcgIts

  t1 = time()

  la = lap(a)

  v = vec(sum(a,dims=1))
  v = v .* (1 .+ rand(length(v)))
  p = sortperm(v)

  llmat = LLMatOrd(a,p)
  ldli = approxChol(llmat)

  ip = invperm(p)
  ldlip = LDLinv(p[ldli.col], ldli.colptr, p[ldli.rowval], ldli.fval, ldli.d[ip]);

  F = function(b)
    x = zeros(size(b))
    x = LDLsolver(ldlip, b)
    #x[p] = LDLsolver(ldli, b[p])
    return x
  end

  if verbose
    println("Using wted degree ordering. Factorization time: ", time()-t1)
    println("Ratio of operator edges to original edges: ", 2 * length(ldli.fval) / nnz(a))
  end

  if verbose
      println("ratio of max to min diagonal of laplacian : ", maximum(diag(la))/minimum(diag(la)))
  end


  f(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) = pcg(la, b .- mean(b), F, tol=tol, maxits=maxits, maxtime=maxtime, pcgIts=pcgIts, verbose=verbose, stag_test = params.stag_test)

end



function approxchol_lap1(a::SparseMatrixCSC{Tv,Ti};
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams()) where {Tv,Ti}

    tol_ =tol
    maxits_ =maxits
    maxtime_ =maxtime
    verbose_ =verbose
    pcgIts_ =pcgIts


    if params.order == :deg

      return approxchol_lapGreedy(a,
        verbose=verbose,
        tol=tol,
        maxits=maxits,
        maxtime=maxtime,
        pcgIts=pcgIts,
        params=params)


    elseif params.order == :wdeg

      return approxchol_lapWdeg(a,
        verbose=verbose,
        tol=tol,
        maxits=maxits,
        maxtime=maxtime,
        pcgIts=pcgIts,
        params=params)


    else
      return approxchol_lapGiven(a,
        verbose=verbose,
        tol=tol,
        maxits=maxits,
        maxtime=maxtime,
        pcgIts=pcgIts,
        params=params)


    end

end

"""
    solver = approxchol_sddm(sddm); x = solver(b);
    solver = approxchol_sddm(sddm; tol=1e-6, maxits=1000, maxtime=Inf, verbose=false, pcgIts=Int[], params=ApproxCholParams())

Solves sddm systems by wrapping approxchol_lap.
Not yet optimized directly for sddm.

For more info, see http://danspielman.github.io/Laplacians.jl/latest/usingSolvers/index.html
"""
approxchol_sddm = sddmWrapLap(approxchol_lap)




#===============================

  Checking the condition number

=================================#

"""
    cn = condNumber(a, ldli; verbose=false)

Given an adjacency matrix a and an ldli computed by approxChol,
this computes the condition number.
"""
function condNumber(a, ldli; verbose=false)
  la = lap(a)

  # construct the square operator
  g = function(b)

    y = copy(b)

    #=
    mu = mean(y)
    @inbounds for i in eachindex(y)
        y[i] = y[i] - mu
    end
      =#

    @inbounds for i in 1:(length(ldli.d))
        if ldli.d[i] != 0
            y[i] /= (ldli.d[i])^(1/2)
        else
            y[i] = 0
        end
    end

    backward!(ldli, y)

    y = la * y

    forward!(ldli, y)

    @inbounds for i in 1:(length(ldli.d))
        if ldli.d[i] != 0
            y[i] /= (ldli.d[i])^(1/2)
        else
            y[i] = 0
        end
    end

    #=
    mu = mean(y)
    @inbounds for i in eachindex(y)
        y[i] = y[i] - mu
    end
    =#

    return y
  end

  gOp = SqLinOp(true,1.0,size(a,1),g)
  upper = eigs(gOp;nev=1,which=:LM,tol=1e-2)[1][1]

  g2(b) = upper*b - g(b)
  g2Op = SqLinOp(true,1.0,size(a,1),g2)
  lower = upper - eigs(g2Op;nev=2,which=:LM,tol=1e-2)[1][2]

  if verbose
      println("lower: ", lower, ", upper: ", upper);
  end

  return upper/lower

end



#===========================================

  Alternate solver approach

===========================================#


"""
    L = ldli2Chol(ldli)
This produces a matrix L so that L L^T approximate the original Laplacians.
It is not quite a Cholesky factor, because it is off by a perm
(and the all-1s vector orthogonality.
"""
function ldli2Chol(ldli)
    n = length(ldli.colptr)
    m = n + length(ldli.fval)
    li = zeros(Int,m)
    lj = zeros(Int,m)
    lv = zeros(Float64,m)
    lptr = 0

    dhi = zeros(n)
    for i in 1:n
        if ldli.d[i] == 0
            dhi[i] = 1.0
        else
            dhi[i] = sqrt(ldli.d[i])
        end
    end

    scales = ones(n)
    for ii in 1:(n-1)
        i = ldli.col[ii]
        j0 = ldli.colptr[ii]
        j1 = ldli.colptr[ii+1]-1
        scales[i] = prod(1.0 .- ldli.fval[j0:(j1-1)])
    end

    for ii in 1:(n-1)
        i = ldli.col[ii]
        j0 = ldli.colptr[ii]
        j1 = ldli.colptr[ii+1]-1
        scale = scales[i] / dhi[i]

        scj = 1
        for jj in j0:(j1-1)
            j = ldli.rowval[jj]
            f = ldli.fval[jj]

            lptr += 1
            li[lptr] = i
            lj[lptr] = j
            lv[lptr] = -f*scj/scale


            scj = scj*(1-f)
        end
        j = ldli.rowval[j1]

        lptr += 1
        li[lptr] = i
        lj[lptr] = j
        lv[lptr] = -dhi[i]

        lptr += 1
        li[lptr] = i
        lj[lptr] = i
        lv[lptr] = 1/scale

    end

    for i in 1:n
        if ldli.d[i] == 0
            lptr += 1
            li[lptr] = i
            lj[lptr] = i
            lv[lptr] = 1.0
        end
    end

    return sparse(li,lj,lv,n,n)
    #return li, lj, lv
end

function LDLsolver(L::SparseMatrixCSC, b::Array)
    y = x6 = L \ (L' \ b)
    return y .- mean(y)
end


"""
This variation of approxChol creates a cholesky factor to do the elimination.
It has not yet been optimized, and does not yet make the cholesky factor lower triangular
"""
function approxchol_lapChol(a::SparseMatrixCSC{Tv,Ti}; tol::Real=1e-6, maxits=1000, maxtime=Inf, verbose=false, pcgIts=Int[]) where {Tv,Ti}

    tol_ =tol
    maxits_ =maxits
    maxtime_ =maxtime
    verbose_ =verbose
    pcgIts_ =pcgIts

    t1 = time()
    llmat = LLmatp(a)

    ldli = approxChol(llmat)

    chL = ldli2Chol(ldli)

    if verbose
      println("Factorization time: ", time()-t1)
      println("Ratio of operator edges to original edges: ", 2 * length(ldli.fval) / nnz(a))
    end

    F(b) = LDLsolver(chL, b)

    la = lap(a)

    if verbose
        println("ratio of max to min diagonal of laplacian : ", maximum(diag(la))/minimum(diag(la)))
    end


    f(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) = pcg(la, b .- mean(b), F, tol=tol, maxits=maxits, maxtime=maxtime, pcgIts=pcgIts, verbose=verbose)

    return f
end




#=============================================================

ApproxCholPQ
It only implements pop, increment key, and decrement key.
All nodes with degrees 1 through n appear in their own doubly-linked lists.
Nodes of higher degrees are bundled together. (((((( but how is this possible, weighted graph?)

=============================================================#


function keyMap(x, n)
    return x <= n ? x : n + div(x,n)
end
# input is a degree vector, this function builds a priority queue
function ApproxCholPQ(a::Vector{Tind}) where Tind

    n = length(a)
    elems = Array{ApproxCholPQElem{Tind}}(undef, n) # objects in the linked list
    lists = zeros(Tind, 2*n+1) # the head (idx in elems) of doubly-linked lists of degree k, but why 2*n + 1??
    minlist = one(n) # just 1 of the same type as n
    deg1 = 0

    for i in 1:length(a) # i here is the node. This loop iterate through all the nodes in a graph
        key = a[i] # key is the degree
        if key == 1
          deg1 += 1 # keep track of nodes with degree 1
        end
        head = lists[key]

        if head > zero(Tind) # if already has key, (i) -> (head)
            elems[i] = ApproxCholPQElem{Tind}(zero(Tind), head, key)

            elems[head] = ApproxCholPQElem{Tind}(i, elems[head].next, elems[head].key)
        else # just create a node, do not link
            elems[i] = ApproxCholPQElem{Tind}(zero(Tind), zero(Tind), key)

        end

        lists[key] = i # replace head with new head
    end

    return ApproxCholPQ(elems, lists, minlist, n, n, deg1)
end

function approxCholPQPop!(pq::ApproxCholPQ{Tind}) where Tind
    if pq.nitems == 0
        error("ApproxPQ is empty")
    end
    # the pop will try to get an node from low degree to high degree
    while pq.lists[pq.minlist] == 0 # current min degree doubly-linked list is exhausted, try next
        pq.minlist = pq.minlist + 1
    end
    i = pq.lists[pq.minlist] # the start idx in elems of the head of the linked list
    next = pq.elems[i].next


    pq.lists[pq.minlist] = next # pop the head of the doubly-linked list & reset a head
    if next > 0 # if the next element is valid, then reset the prev pointer of that node
        pq.elems[next] = ApproxCholPQElem(zero(Tind), pq.elems[next].next, pq.elems[next].key)
    end

    pq.nitems -= 1 # we poped an element

    if pq.elems[i].key == 1
      pq.deg1 -= 1 # a degree 1 node is consumed
    end

    return i # return the idx of that element
end

function approxCholPQMove!(pq::ApproxCholPQ{Tind}, i, newkey, oldlist, newlist) where Tind

    prev = pq.elems[i].prev
    next = pq.elems[i].next

    # remove i from its old list
    if next > zero(Tind)
        pq.elems[next] = ApproxCholPQElem{Tind}(prev, pq.elems[next].next, pq.elems[next].key)
    end
    if prev > zero(Tind)
        pq.elems[prev] = ApproxCholPQElem{Tind}(pq.elems[prev].prev, next, pq.elems[prev].key)

    else
        pq.lists[oldlist] = next
    end

    # insert i into its new list
    head = pq.lists[newlist]
    if head > 0
        pq.elems[head] = ApproxCholPQElem{Tind}(i, pq.elems[head].next, pq.elems[head].key)
    end
    pq.lists[newlist] = i

    pq.elems[i] = ApproxCholPQElem{Tind}(zero(Tind), head, newkey)

    return nothing
end

"""
    Decrement the key of element i
    This could crash if i exceeds the maxkey
"""
function approxCholPQDec!(pq::ApproxCholPQ{Tind}, i) where Tind

    oldlist = keyMap(pq.elems[i].key, pq.n)
    newlist = keyMap(pq.elems[i].key - one(Tind), pq.n)
    if pq.elems[i].key - one(Tind) == 1
      pq.deg1 += 1 # a node is reduce to degree 1
    end

    if newlist != oldlist

        approxCholPQMove!(pq, i, pq.elems[i].key - one(Tind), oldlist, newlist)

        if newlist < pq.minlist
            pq.minlist = newlist
        end

    else
        pq.elems[i] = ApproxCholPQElem{Tind}(pq.elems[i].prev, pq.elems[i].next, pq.elems[i].key - one(Tind))
    end


    return nothing
end

"""
    Increment the key of element i
    This could crash if i exceeds the maxkey
"""
function approxCholPQInc!(pq::ApproxCholPQ{Tind}, i) where Tind

    oldlist = keyMap(pq.elems[i].key, pq.n)
    newlist = keyMap(pq.elems[i].key + one(Tind), pq.n)
    # the node with degree = 1 will not have other edges

    if newlist != oldlist

        approxCholPQMove!(pq, i, pq.elems[i].key + one(Tind), oldlist, newlist)

    else
        pq.elems[i] = ApproxCholPQElem{Tind}(pq.elems[i].prev, pq.elems[i].next, pq.elems[i].key + one(Tind))
    end

    return nothing
end
