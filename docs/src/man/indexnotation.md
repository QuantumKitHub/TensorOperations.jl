# Index notation with macros

The main export and main functionality of TensorOperations.jl is the `@tensor` macro

```@docs
@tensor
```

The functionality and configurability of `@tensor` and some of its relatives is explained in
detail on this page.

## The `@tensor` macro

The preferred way to specify (a sequence of) tensor operations is by using the `@tensor`
macro, which accepts an
[index notation](https://en.wikipedia.org/wiki/Abstract_index_notation) format, a.k.a.
[Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) (and in particular,
Einstein's summation convention).

This can most easily be explained using a simple example:

```julia
using TensorOperations
α = randn()
A = randn(5, 5, 5, 5, 5, 5)
B = randn(5, 5, 5)
C = randn(5, 5, 5)
D = zeros(5, 5, 5)
@tensor begin
    D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
end
```

The first important observation is the use of two different assignment operators within the
body of the `@tensor` call. The regular assignment operator `=` stores the result of the
tensor expression in the right hand side in an existing tensor `D`, whereas the 'definition'
operator `:=` results in a new tensor `E` with the correct properties to be created.
Nonetheless, the contents of `D` and `E` will be equal.

!!! warning "Aliasing"

    Special care has to be taken when using the in-place `=`, combined with tensors that might
    alias or have overlapping memory regions. As `@tensor` assumes it is allowed to change the
    order of the operations within its scope, this may lead to unexpected or wrong results.
    For example, the result of the following expressions might be different and making use of
    this should be considered as undefined behavior:
    ```julia
    @tensor B[:] := L[-1, 1] * A[1, -2, 2] * R[2, -3] + α * A[-1, -2, -3]
    @tensor A[:] = L[-1, 1] * A[1, -2, 2] * R[2, -3] + α * A[-1, -2, -3]
    ```

Following Einstein's summation convention, that contents is computed in a number of steps
involving the three primitive tensor operators. In this particular example, the first step
involves tracing/contracting the 3rd and 5th index of array `A`, the result of which is
stored in a temporary array which thus needs to be created. This resulting array will then
be contracted with array `B` by contracting its 2nd index with the last index of `B` and its
last index with the first index of `B`. The result is stored in `D` in the first line, or in
a newly allocated array which will end up being `E` in the second line. Note that the index
order of `D` and `E` is such that its first index corresponds to the first index of `A`, the
second index corresponds to the second index of `B`, whereas the third index corresponds to
the fourth index of `A`. Finally, the array `C` (scaled with `α`) is added to this result
(in place), which requires a further index permutation.

The index pattern is analyzed at compile time and expanded to a set of calls to the basic
tensor operations, i.e. [`tensoradd!`](@ref), [`tensortrace!`](@ref) and
[`tensorcontract!`](@ref). Temporaries are created where necessary, as these building blocks
operate pairwise on the input tensors. The generated code can easily be inspected

```@example
using TensorOperations
@macroexpand @tensor E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
```

The different functions in which this tensor expression is decomposed are discussed in more
detail in the [Implementation](@ref) section of this manual.

In this example, the tensor indices were labeled with arbitrary letters; also longer names
could have been used. In fact, any proper Julia variable name constitutes a valid label.
Note though that these labels are never interpreted as existing Julia variables. Within the
`@tensor` macro they are converted into symbols and then used as dummy names, whose only
role is to distinguish the different indices. Their specific value bears no meaning. They
also do not appear in the generated code as illustrated above. This implies, in particular,
that the specific tensor operations defined by the code inside the `@tensor` environment are
completely specified at compile time. Various remarks regarding the index notation are in
order.

1.  TensorOperations.jl only supports strict Einstein summation convention. This implies
    that there are two types of indices. Either an index label appears once in every term of
    the right hand side, and it also appears on the left hand side. We refer to the
    corresponding indices as *open or free*. Alternatively, an index label appears exactly
    twice within a given term on the right hand side. The corresponding indices are referred
    to as *closed or contracted*, i.e. the pair of indices takes equal values and are summed
    over their (equal) range. This is known as a contraction, either an outer contraction
    (between two indices of two different tensors) or an inner contraction (a.k.a. trace,
    between two indices of a single tensor). More liberal use of the index notation, such as
    simultaneous summutation over three or more indices, or a open index appearing
    simultaneously in different tensor factors, are not supported by TensorOperations.jl.

2.  Aside from valid Julia identifiers, index labels can also be specified using literal
    integer constants or using a combination of integers and symbols. Furthermore, it is
    also allowed to use primes (i.e. Julia's `adjoint` operator) to denote different
    indices, including using multiple subsequent primes. The following expression thus
    computes the same result as the example above:

   ```julia
   @tensor D[å'', ß, clap'] = A[å'', 1, -3, clap', -3, 2] * B[2, ß, 1] + α * C[clap', å'', ß]
   ```

3.  If only integers are used for specifying index labels, this can be used to control the
    pairwise contraction order, by using the well-known NCON convention, where open indices
    in the left hand side are labelled by negative integers `-1`, `-2`, `-3`, whereas
    contraction indices are labelled with positive integers `1`, `2`, … Since the index
    order of the left hand side is in that case clear from the right hand side expression,
    the left hand side can be indexed with `[:]`, which is automatically replaced with all
    negative integers appearing in the right hand side, in decreasing order. The value of
    the labels for the contraction indices determines the pairwise contraction order. If
    multiple tensors need to be contracted, a first temporary will be created consisting of
    the contraction of the pair of tensors that share contraction index `1`, then the pair
    of tensors that share contraction index `2` (if not contracted away in the first pair)
    will be contracted, and so forth. The next subsection explains contraction order in more
    detail and gives some useful examples, as the example above only includes a single pair
    of tensors to be contracted.

4.  Index labels always appear in square brackets `[ ... ]` but can be separated by either
    commas, as in `D[a, b, c]`, (yielding a `:ref` expression) or by spaces, as in
    `D[a b c]`, (yielding a `:typed_hcat` expression).
    
    There is also the option to separate the indices into two groups using a semicolon. This
    can be useful for tensor types which have two distinct set of indices, but has no effect
    when using Julia `AbstractArray` objects. While in principle both spaces and commas can
    be used within the two groups, e.g. as in `D[a, b; c]` or `D[a b; c]`, there are some
    restrictions because of accepted Julia syntax. Both groups of indices should use the
    same convention. If there is only a single index in the first group, the second group
    should use spaces to constitute a valid expression. Finally, having no indices in the
    first group is only possible by writing an empty tuple. The second group can then use
    spaces, or also contain the indices as a tuple, i.e. both `D[(); a b c]` or `D[(); (a,
    b, c)]`. Writing the two groups of indices within a tuple (which uses a comma as natural
    separator), with both tuples seperated by a semicolon is always valid syntax,
    irrespective of the number of indices in that group.

5.  Index expressions `[...]` are only interpreted as index notation on the highest level.
    For example, if you want to mulitply two matrices which are stored in a list, you can
    write
    
   ```julia
   @tensor result[i,j] := list[1][i,k] * list[2][k,j]
   ```

    However, if both are stored as a the slices of a 3-way array, you cannot write

   ```julia
   @tensor result[i,j] := list[i,k,1] * list[k,j,2]
   ```

    Rather, you should use

   ```julia
   @tensor result[i,j] := list[:,:,1][i,k] * list[:,:,2][k,j]
   ```

    or, if you want to avoid additional allocations

   ```julia
   @tensor result[i,j] := view(list,:,:,1)[i,k] * view(list,:,:,2)[k,j]
   ```

Note, finally, that the `@tensor` specifier can be put in front of a single tensor
expression, or in front of a `begin ... end` block to group and evaluate different
expressions at once. Within an `@tensor begin ... end` block, the `@notensor` macro can be
used to annotate indexing expressions that need to be interpreted literally. 

```@docs
@notensor
```

As in illustration, note that the previous code examples about the matrix multiplication
with matrices stored in a 3-way array can now also be written as

```julia
@tensor begin
    @notensor A = list[:,:,1]
    @notensor B = list[:,:,2]
    result[i,j] = A[i,k] * B[k,j]
end
```

## Contraction order specification and optimisation

A contraction of several (more than two) tensors, as in

```julia
@tensor D[a, d, j, i, g] := A[a, b, c, d, e] * B[b, e, f, g] * C[c, f, i, j]
```

is known as a *tensor network* and is generically evaluated as a sequence of pairwise
contractions. In the example above, this contraction is evaluated using Julia's default left
to right order, i.e. as `(A[a, b, c, d, e] * B[b, e, f, g]) * C[c, f, i, j]`. There are
however different strategies to modify this order.

1.  Explicit parenthesis can be used to group subnetworks within a tensor network that will
    be evaluated first. Parentheses around subexpressions are always respected by the
    `@tensor` macro.

2.  As explained in the previous subsection, if one respects the
    [NCON](https://arxiv.org/abs/1402.0939) convention of specifying indices, i.e. positive
    integers for the contracted indices and negative indices for the open indices, the
    different factors will be reordered and so that the pairwise tensor contractions
    contract over indices with smaller integer label first. For example,

   ```julia
   @tensor D[:] := A[-1, 3, 1, -2, 2] * B[3, 2, 4, -5] * C[1, 4, -4, -3]
   ```

    will be evaluated as `(A[-1, 3, 1, -2, 2] * C[1, 4, -4, -3]) * B[3, 2, 4, -5]`.
    Furthermore, in that case the indices of the output tensor (`D` in this case) do not
    need to be specified (using `[:]` instead), and will be chosen as
    `(-1, -2, -3, -4, -5)`. Note that if two tensors are contracted, all contraction indices
    among them will be contracted, even if there are additional contraction indices whose
    label is a higher positive number. For example,
    
   ```julia
   @tensor D[:] := A[-1, 3, 2, -2, 1] * B[3, 1, 4, -5] * C[2, 4, -4, -3]
   ```

    amounts to the original left to right order, because `A` and `B` share the first
    contraction index `1`. When `A` and `B` are contracted, also the contraction with label
    `3` will be performed, even though contraction index with label `2` is not yet
    'processed'.

3.  A specific contraction order can be manually specified by supplying an `order` keyword
    argument to the `@tensor` macro. The value is a tuple of the contraction indices in the
    order that they should be dealt with, e.g. the default order could be changed to first
    contract `A` with `C` using
    
   ```julia
   @tensor order=(c, b, e, f) begin
       D[a, d, j, i, g] := A[a, b, c, d, e] * B[b, e, f, g] * C[c, f, i, j]
   end
   ```

    Here, the same comment as in the NCON style applies; once two tensors are contracted
    because they share an index label which is next in the `order` list, all other indices
    with shared label among them will be contracted, irrespective of their order.

!!! warning "Combining order specifications"

    Note that it is currently not possible to combine the NCON style convention of specifying
    indices with the use of parentheses. If both are used at the same time, the parentheses
    take precedence and the NCON style will be ignored. Any remaining contraction orders
    will be evaluated in the default left to right order. For example, in the following
    contractions, we have `E1 = A * ((B * D) * C)`, but `E2 = A * ((B * C) * D)`. This is
    true even when the parentheses are compatible with the NCON contraction order, as is
    the case here.

    ```julia
    @tensor E1[-1 -2 -3; -4] := A[-1 -2 -3; 4 5] * B[4; 1] * C[5; 2] * D[1 2; -4]
    @tensor E2[-1 -2 -3; -4] := A[-1 -2 -3; 4 5] * (B[4; 1] * C[5; 2] * D[1 2; -4])
    ```

    Additionally, combining the `order = (...)` keyword with parentheses is currently not supported.

In the case of more complex tensor networks, the optimal contraction order cannot always
easily be guessed or determined on plain sight. It is then useful to be able to optimize the
contraction order automatically, given a model for the complexity of contracting the
different tensors in a particular order. This functionality is provided where the cost
function being minimized models the computational complexity by counting the number of
scalar multiplications. This minimisation problem is solved using the algorithm that was
described in
[Physical Review E 90, 033315 (2014)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.033315).
For a given tensor networ contraction, this algorithm is ran once at compile time, while
lowering the tensor espression, and the outcome will be hard coded in the expression
resulting from the macro expansion. While the computational complexity of this optimisation
algorithm scales itself exponentially in the number of tensors involved in the network, it
should still be acceptibly fast (milliseconds up to a few seconds at most) for tensor
network contractions with up to around 30 tensors. Information of the optimization process
can be obtained during compilation by using the alternative macro `@tensoropt_verbose`.

As the cost is determined at compile time, it is not using actual tensor properties (e.g.
`size(A, i)` in the case of arrays) in the cost model, and the cost or extent associated
with every index can be specified in various ways, either using integers or floating point
numbers or some arbitrary univariate polynomial of an abstract variable, e.g. `χ`. In the
latter case, the optimization assumes the asymptotic limit of large `χ`.

```@docs
@tensoropt
@tensoropt_verbose
```

As an final remark, the optimization can also be accessed directly from `@tensor` by
specifying the additional keyword argument `opt=true`, which will then use the default cost
model, or `opt=optex` to further specify the costs.

```julia
# cost χ for all indices (a, b, c, d, e, f)
@tensor opt=true D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]

# cost χ for indices (a, b, c, e), other indices (d, f) have cost 1
@tensor opt=(a, b, c, e) D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]

# cost 1 for indices (a, b, c, e), other indices (d, f) have cost χ
@tensor opt=!(a, b, c, e) D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]

# cost as specified for listed indices, unlisted indices have cost 1 (any symbol for χ can be used)
@tensor opt=(a => χ, b => χ^2, c => 2 * χ, e => 5) begin
    D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
end

```

## Dynamical tensor network contractions with `ncon` and `@ncon`

Tensor network practicioners are probably more familiar with the network contractor function
[`ncon`](@ref) to perform a tensor network contraction, as e.g. described in
[NCON](https://arxiv.org/abs/1402.0939). In particular, a graphical application
[TensorTrace](https://www.tensortrace.com) was recently introduced to facilitate the
generation of such `ncon` calls. TensorOperations.jl provides compatibility with this
interface by also exposing an `ncon` function with the same basic syntax

```julia
ncon(list_of_tensor_objects, list_of_index_lists)
```

e.g. the example of above is equivalent to

```julia
@tensor D[:] := A[-1, 3, 1, -2, 2] * B[3, 2, 4, -5] * C[1, 4, -4, -3]
D ≈ ncon((A, B, C), ([-1, 3, 1, -2, 2], [3, 2, 4, -5], [1, 4, -4, -3]))
```

where the lists of tensor objects and of index lists can be given as a vector or a tuple.
The `ncon` function necessarily needs to analyze the contraction pattern at runtime, but
this can be an advantage, in cases where the contraction is determined by runtime
information and thus not known at compile time. A downside from this, besides the fact that
this can result in some overhead (though this is typically negligable for anything but very
small tensor contractions), is that `ncon` is type-unstable, i.e. its return type cannot be
inferred by the Julia compiler.

The full call syntax of the `ncon` method exposed by TensorOperations.jl is

```julia
ncon(tensorlist, indexlist, [conjlist]; order=..., output=...)
```

where the first two arguments are those of above. Let us first discuss the keyword
arguments. The keyword argument `order` can be used to change the contraction order, i.e. by
specifying which contraction indices need to be processed first, rather than the strictly
increasing order `[1, 2, ...]`, as discussed in the previous subsection. The keyword
argument `output` can be used to specify the order of the output indices, when it is
different from the default `[-1, -2, ...]`.

The optional positional argument `conjlist` is a list of `Bool` variables that indicate
whether the corresponding tensor needs to be conjugated in the contraction. So while

```julia
ncon([A, conj(B), C], [[-1, 3, 1, -2, 2], [3, 2, 4, -5], [1, 4, -4, -3]]) ≈
ncon([A, B, C], [[-1, 3, 1, -2, 2], [3, 2, 4, -5], [1, 4, -4, -3]], [false, true, false])
```

the latter has the advantage that conjugating `B` is not an extra step (which creates an
additional temporary requiring allocations), but is performed at the same time when it is
contracted.

As an alternative solution to the optional positional arguments, there is also an `@ncon`
macro. It is just a simple wrapper over an `ncon` call and thus does not analyze the
indices at compile time, so that they can be fully dynamical. However, it will transform

```julia
@ncon([A, conj(B), C], indexlist; order=..., output=...)
```

into

```julia
ncon(Any[A, B, C], indexlist, [false, true, false]; order=..., output=...)
```

so as to get the advantages of just-in-time conjugation (pun intended) using the familiar
looking `ncon` syntax.

```@docs
ncon
@ncon
```

## Index compatibility and checks

Errors in the index pattern are caught at compile time, and will spawn an error of type
```@docs
TensorOperations.IndexError
```

Once the expression is compiled, there can still be errors that will lead to runtime errors,
the type of which depends on the specific type of tensor (e.g. `DimensionMismatch` for
`AbstractArray` objects). Indeed, indices with the same label, either open indices on the 
two sides of the equation, or contracted indices, need to be compatible. For `AbstractArray` 
objects, this means they must have the same size. Other tensor types might have more complicated 
structure associated with their indices, and requires matching between those. The function
[`checkcontractible`](@ref) is part of the interface that can be used to control when
tensors can be contracted with each other along specific indices.

For example, for `AbstractArray` objects, the following helper methods are defined that 
capture the recurring checks across different implementations of the tensor operations:

```@docs
TensorOperations.argcheck_index2tuple
TensorOperations.argcheck_tensoradd
TensorOperations.argcheck_tensortrace
TensorOperations.argcheck_tensorcontract
TensorOperations.dimcheck_tensoradd
TensorOperations.dimcheck_tensortrace
TensorOperations.dimcheck_tensorcontract
```

Whenever an incompatible index structures is detected at runtime, the resulting error will
originate deep within the implementation, at which point the error message will provide 
little information as to which specific tensors and which indices are producing the mismatch. 
When debugging, it might be useful to add the keyword argument `contractcheck = true` to 
the `@tensor` macro. Explicit checks using `checkcontractible` are then enabled that are run 
before any tensor operation is performed. When a mismatch is detected, these checks still 
have access to the label information and spawn a more informative error message.

A different type of check is the `costcheck` keyword argument, which can be given the values
`:warn` or `:cache`. With either of both values for this keyword argument, additional checks
are inserted that compare the contraction order of any tensor contraction of three or more
factors against the optimal order based on the current tensor size. More generally, the
function [`tensorcost`](@ref) is part of the interface and associated a cost value with
every index of a tensor, which is then used in the cost model. With `costcheck=:warn`, a
warning will be spawned for every tensor network where the actual contraction order (even
when optimized using abstract costs) does not match with the ideal contraction order given
the current `tensorcost` values. With `costcheck = :cache`, the tensor networks with
non-optimal contraction order are stored in a global package variable
`TensorOperations.costcache`. However, when a tensor network is evaluated several times with
different tensor sizes or tensor costs, only the evaluation giving rise to the largest total
contraction cost for that network will appear in the cache (provided the actual contraction
order deviates from the optimal order in that largest case).

## Backends, multithreading and GPUs

Every index expression will be evaluated as a sequence of elementary tensor operations, i.e.
permuted additions, partial traces and contractions, which are implemented for strided
arrays as discussed in [Package features](@ref). In particular, optimised implementations rely
on [Strided.jl](https://github.com/Jutho/Strided.jl), and we refer to this package for a
full specification of which arrays are supported. As a rule of thumb, this primarily
includes `Array`s from Julia base, as well as `view`s thereof if sliced with a combination
of `Integer`s and `Range`s. Special types such as `Adjoint` and `Transpose` from Base are
also supported. For permuted addition and partial traces, native Julia implementations are
used which could benefit from multithreading if `JULIA_NUM_THREADS>1`.

The binary contraction is performed by first permuting the two input tensors into a form
such that the contraction becomes equivalent to one matrix multiplication on the whole data,
followed by a final permutation to bring the indices of the output tensor into the desired
order. This approach allows to use the highly efficient matrix multiplication kernel
(`gemm`) from BLAS, which is multithreaded by default. There is also a native contraction
implementation that is used for e.g. arrays with an `eltype` that is not
`<:LinearAlgebra.BlasFloat`. It performs the contraction directly without the additional
permutations, but still in a cache-friendly and multithreaded way (again relying on
`JULIA_NUM_THREADS > 1`). This implementation can also be used for `BlasFloat` types (but
will typically be slower), and the use of BLAS can be controlled by explicitly switching the
backend between `StridedBLAS` and `StridedNative` using the `backend` keyword to
[`@tensor`](@ref). Similarly, different allocation strategies, when available, can be
selected using the `allocator` keyword of [`@tensor`](@ref).

Since TensorOperations v5, there are also fallback implementations for non-strided arrays
that only use functionality from `Base`, as well as `LinearAlgebra.mul!`. The manual section
[Backends and Allocators](@ref) provides a more elaborate discussion.

The primitive tensor operations are also implemented for `CuArray` objects of the
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) library. This implementation is essentially a
simple wrapper over the cuTENSOR library of NVidia, and will only be loaded when the
`cuTENSOR.jl` package is loaded. The `@tensor` macro will then automatically work for
operations between GPU arrays.

Mixed operations between host arrays (e.g. `Array`) and device arrays (e.g. `CuArray`) will
fail. However, if one wants to harness the computing power of the GPU to perform all tensor
operations, there is a dedicated macro `@cutensor`. This will transfer all host arrays to
the GPU before performing the requested operations. If the output is an existing host array,
the result will be copied back. If a new result array is created (i.e. using `:=`), it will
remain on the GPU device and it is up to the user to transfer it back. Arrays are transfered
to the GPU just before they are first used, and in a complicated tensor expression, this
might have the benefit that transer of the later arrays overlaps with computation of earlier
operations.

```@docs
@cutensor
```
