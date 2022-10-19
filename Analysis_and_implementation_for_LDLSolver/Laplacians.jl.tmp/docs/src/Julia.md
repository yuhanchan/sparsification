# Using Julia

These are some things you might want to know about using Julia if it is new to you.  There are now many other resources that can explain Julia to you.  But, we keep this section here for reference.

## Docstrings

Julia 0.5 lets you take advantage of docstrings.
For example, `?ringGraph` produces

~~~
The simple ring on n vertices
~~~

When having a multiline comment, make sure that lines don't have starting and trailing spaces.
This will mess up the indentation when calling '?func_name'.

## Julia Notebooks
To get the Julia notebooks working, I presently type `jupyter notebook`.
I then select the kernel to be Julia-0.5.0.
It seems important to run this command from a directory that contains all the directories
that have notebooks that you will use.  In particular, I advise against "uploading" notebooks
from other directories.  That has only given me trouble.



To turn a notebook into html, you type something like

~~~
jupyter nbconvert Laplacians.ipynb
~~~

or

~~~
jupyter nbconvert --to markdown --stdout Sampler.ipynb > SamplerNotebook.md
~~~

## Workflows

Julia has an IDE called Juno.  Both Dan and Serban have encountered some trouble with it: we have both found that it sometimes refuses to reload .jl code that we have written.  Please document workflows that you have found useful here:

#### Dan's current workflow:
* I use emacs (which has a mode for Julia) and the notebooks.
* I develop Julia code in a "temporary" file with a name like develX.jl.  While I am developing, this code is not included by the module to which it will eventually belong.
* After modifying code, I reload it with `include("develX.jl")`.  This works fine for reloading methods.  It is not a good way to reload modules or types.  So, I usually put the types either in a separate file, or in my julia notebook.

* I am writing this documention in MacDown.

#### Add your current workflow here:

## Things to be careful of (common bugs)

* Julia passes vectors and matrices to routines by reference, rather than by copying them.  If you type `x = y` when x and y are arrays, then this will make x a pointer to y.  If you want x to be a copy of y, type `x = copy(y)`.  This can really mess up matlab programmers.  I wrote many functions that were modifying their arguments without realizing it.

* On the other hand, if you type `x = x + y`, then x becomes a newly allocated vector and no longer refers to the original.  This is true even if you type `x += y`.  Here is an example that shows two of the possible behaviors, and the difference between what happens inside functions.

~~~julia

"Adds b in to a"
function addA2B(a,b)
    for i in 1:length(a)
        a[i] += b[i]
    end
end

"Fails to add b in to a"
function addA2Bfail(a,b)
	a += b
end

a = [1 0]
b = [2 2]
addA2B(a,b)
a

1x2 Array{Int64,2}:
 3  2

a = [1 0]
b = [2 2]
addA2Bfail(a,b)
a

1x2 Array{Int64,2}:
 1  0

a += b
a

1x2 Array{Int64,2}:
 3  2

~~~

* If you are used to programming in Matlab, you might be tempted to type a line like `for i in 1:10,`.  _Do not put extra commas in Julia!_  It will cause bad things to happen.

* To get a vector with entries 1 through n, type `collect(1:n)`.  The object `1:n` is a range, rather than a vector.

* Julia sparse matrix entries dissapear if they are set to 0. In order to overcome this, use the `setValue` function. `setValue(G, u, i, 0)` will set `weighti(G, u, i)` to 0 while also leaving `(u, nbri(G, u, i))` in the matrix.  _Note This problem may have been fixed with Julia version 0.5._

## Useful Julia functions

I am going to make a short list of Julia functions/features that I find useful.  Please add those that you use often as well.

* docstrings: in the above example, I used a docstring to document each function.  You can get these by typing `?addA2B`.  You can also  [write longer docstrings and use markdown](http://julia.readthedocs.org/en/latest/manual/documentation/).  I suggest putting them in front of every function.

* `methods(foo)` lists all methods with the name foo.
* `fieldnames(footype)` tells you all the fields of footype.  Note that this is 0.4.  In 0.3.11, you type `names(footype)`

~~~julia
julia> a = sparse(rand(3,3));
julia> fieldnames(a)
5-element Array{Symbol,1}:
 :m
 :n
 :colptr
 :rowval
 :nzval
~~~


## Optimizing code in Julia

The best way that I've found of figuring out what's slowing down my code has been to use `@code_warntype`.  

Note that the first time you run a piece of code in Julia, it gets compiled.  So, you should run it on a small example before trying to time it.  Then, use `@time` to time your code.

I recommend reading the Performance Tips in the Julia documentation, not that I've understood all of it yet.




## How should notebooks play with Git?

The great thing about the notebooks is that they contain live code, so that you can play with them.  But, sometimes you get a version that serves as great documentation, and you don't want to klobber it my mistake later (or evern worse, have someone else klobber it).  Presumably if someone accidently commits a messed up version we can unwind that.  But, is there a good way to keep track of this?
