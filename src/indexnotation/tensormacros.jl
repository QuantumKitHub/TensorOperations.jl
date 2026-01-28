const defaultparser = TensorParser()

"""
    @notensor(block)

Marks a block which should be ignored within an `@tensor` environment. Has no effect outside of `@tensor`.
"""
macro notensor(ex::Expr)
    return esc(ex)
end

"""
    @tensor(tensor_expr; kwargs...)
    @tensor [kw_expr...] tensor_expr

Specify one or more tensor operations using Einstein's index notation. Indices can
be chosen to be arbitrary Julia variable names, or integers. When contracting several
tensors together, this will be evaluated as pairwise contractions in left to right
order, unless the so-called NCON style is used (positive integers for contracted
indices and negative indices for open indices).

Additional keyword arguments may be passed to control the behavior of the parser:

- `order`: 
    A list of contraction indices of the form `order=(...,)` which specify the order in which they will be contracted.
- `opt`:
    Contraction order optimization, similar to [`@tensoropt`](@ref). Can be either a boolean or an `OptExpr`.
- `contractcheck`:
    Boolean flag to enable runtime check for contractibility of indices with clearer error messages.
- `costcheck`:
    Can be either `:warn` or `:cache` and adds runtime checks to compare the compile-time contraction order to the optimal order computed for the actual run time tensor costs.
    If `costcheck == :warn`, warnings are printed for every sub-optimal contraction that is encountered.
    If `costcheck == :cache`, only the most costly run of a particular sub-optimal contraction will be cached in `TensorOperations.costcache`.
    In both cases, a suggestion for the `order` keyword argument is computed to switch to the optimal contraction order.
- `backend`: 
    Inserts an implementation backend as a final argument in the different tensor operation calls in the generated code.
- `allocator`:
    Inserts an allocation strategy as a final argument in the tensor allocation calls in the generated code.
"""
macro tensor(args::Vararg{Expr})
    isempty(args) && throw(ArgumentError("No arguments passed to `@tensor`"))
    length(args) == 1 && return esc(defaultparser(args[1]))

    tensorexpr = args[end]
    kwargs = parse_tensor_kwargs(args[1:(end - 1)])
    parser = tensorparser(tensorexpr, kwargs...)
    return esc(parser(tensorexpr))
end

function tensorparser(tensorexpr, kwargs...)
    parser = TensorParser()
    # the order of backend and allocator are important so let's find them first
    hasbackend = false
    for (name, val) in kwargs
        if name == :backend
            hasbackend = true
            backend = val
            push!(parser.postprocessors, ex -> insertbackend(ex, backend))
            break
        end
    end
    for (name, val) in kwargs
        if name == :allocator
            allocator = val
            if !hasbackend
                backend = Expr(:call, GlobalRef(TensorOperations, :DefaultBackend))
                push!(parser.postprocessors, ex -> insertbackend(ex, backend))
            end
            push!(parser.postprocessors, ex -> insertallocator(ex, allocator))
            push!(parser.postprocessors, ex -> insertcheckpoints(ex, allocator))
            break
        end
    end
    # now handle the remaining keyword arguments
    for (name, val) in kwargs
        if name == :order
            isexpr(val, :tuple) ||
                throw(ArgumentError("Invalid use of `order`, should be `order=(...,)`"))
            indexorder = map(normalizeindex, val.args)
            parser.contractiontreebuilder = network -> indexordertree(network, indexorder)
        elseif name == :contractcheck
            val isa Bool ||
                throw(ArgumentError("Invalid use of `contractcheck`, should be `contractcheck=bool`."))
            val && push!(parser.preprocessors, ex -> insertcontractionchecks(ex))
        elseif name == :costcheck
            val in (:warn, :cache) ||
                throw(ArgumentError("Invalid use of `costcheck`, should be `costcheck=warn` or `costcheck=cache`"))
            parser.contractioncostcheck = val
        elseif name == :opt
            if val isa Bool && val
                optdict = optdata(tensorexpr)
            elseif val isa Expr
                optdict = optdata(val, tensorexpr)
            else
                throw(ArgumentError("Invalid use of `opt`, should be `opt=true` or `opt=OptExpr`"))
            end
            parser.contractiontreebuilder = network -> optimaltree(network, optdict)[1]
        elseif !(name == :backend || name == :allocator) # these two have been handled
            throw(ArgumentError("Unknown keyword argument `name`."))
        end
    end
    return parser
end

function parse_tensor_kwargs(args)
    # @tensor(tensorexpr; kwargs)
    if length(args) == 1 && isexpr(args[1], :parameters)
        args = args[1].args
    end

    # @tensor kw1=val1 kw2=val2 ... tensorexpr
    function kw_to_pair(ex)
        if (isexpr(ex, :(=), 2) || isexpr(ex, :kw, 2))
            return Pair(ex.args[1], ex.args[2])
        end
        throw(ArgumentError("unknown keyword expression `$ex`"))
    end
    return kw_to_pair.(args)
end

"""
    @tensoropt(optex, block)
    @tensoropt(block)

Specify one or more tensor operations using Einstein's index notation. Indices can
be chosen to be arbitrary Julia variable names, or integers. When contracting several
tensors together, the macro will determine (at compile time) the optimal contraction
order depending on the cost associated to the individual indices. If no `optex` is
provided, all indices are assumed to have an abstract scaling `χ` which is optimized
in the asympotic limit of large `χ`.

The cost can be specified in the following ways:

```julia
# cost χ for all indices (a, b, c, d, e, f)
@tensoropt D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]

# asymptotic cost χ for indices (a, b, c, e), other indices (d, f) have cost 1
@tensoropt (a, b, c, e) C[a, b, c, d] := A[a, e, c, f, h] * B[f, g, e, b] * C[g, d, h]

# cost 1 for indices (a, b, c, e); other indices (d, f) have asymptotic cost χ
@tensoropt !(a, b, c, e) C[a, b, c, d] := A[a, e, c, f, h] * B[f, g, e, b] * C[g, d, h]

# cost as specified for listed indices, unlisted indices have cost 1 (any symbol for χ can be used)
@tensoropt (a => χ, b => χ^2, c => 2 * χ, e => 5) begin
    C[a, b, c, d] := A[a, e, c, f, h] * B[f, g, e, b] * C[g, d, h]
end
```

Note that `@tensoropt` will optimize any tensor contraction sequence it encounters
in the (block of) expressions. It will however not break apart expressions that have
been explicitly grouped with parenthesis, i.e. in
```julia
@tensoropt C[a, b, c, d] := A[a, e, c, f, h] * (B[f, g, e, b] * C[g, d, h])
```
it will always contract `B` and `C` first. For a single tensor contraction sequence,
the optimal contraction order and associated (asymptotic) cost can be obtained using
`@optimalcontractiontree`.
"""
macro tensoropt(expressions...)
    if length(expressions) == 1
        ex = expressions[1]
        optdict = optdata(ex)
    elseif length(expressions) == 2
        optex = expressions[1]
        ex = expressions[2]
        optdict = optdata(optex, ex)
    end

    parser = TensorParser()
    parser.contractiontreebuilder = network -> optimaltree(network, optdict)[1]
    return esc(parser(ex))
end

"""
    @tensoropt_verbose(optex, block)
    @tensoropt_verbose(block)

Similar to [`@tensoropt`](@ref), but prints information details regarding the optimization process to the standard output.
"""
macro tensoropt_verbose(expressions...)
    if length(expressions) == 1
        ex = expressions[1]
        optdict = optdata(ex)
    elseif length(expressions) == 2
        optex = expressions[1]
        ex = expressions[2]
        optdict = optdata(optex, ex)
    end

    parser = TensorParser()
    parser.contractiontreebuilder = network -> optimaltree(network, optdict; verbose = true)[1]
    return esc(parser(ex))
end

macro optimalcontractiontree(expressions...)
    if length(expressions) == 1
        ex = expressions[1]
        optdict = optdata(ex)
    elseif length(expressions) == 2
        optex = expressions[1]
        ex = expressions[2]
        optdict = optdata(optex, ex)
    end

    if isassignment(ex) || isdefinition(ex)
        ex = getrhs(ex)
    end
    if !(ex.head == :call && ex.args[1] == :*)
        error("cannot compute optimal contraction tree for this expression")
    end
    network = [getindices(ex.args[k]) for k in 2:length(ex.args)]
    tree, cost = optimaltree(network, optdict)
    return tree, cost
end

"""
    @ncon(tensorlist, indexlist; order = ..., output = ...)

Contract the tensors in `tensorlist` (of type `Vector` or `Tuple`) according to the network
as specified by `indexlist`. Here, `indexlist` is a list (i.e. a `Vector` or `Tuple`) with
the same length as `tensorlist` whose entries are themselves lists (preferably
`Vector{Int}`) where every integer entry provides a label for corresponding index/dimension
of the corresponding tensor in `tensorlist`. Positive integers are used to label indices
that need to be contracted, and such thus appear in two different entries within
`indexlist`, whereas negative integers are used to label indices of the output tensor, and
should appear only once.

By default, contractions are performed in the order such that the indices being contracted
over are labelled by increasing integers, i.e. first the contraction corresponding to label
`1` is performed. The output tensor had an index order corresponding to decreasing
(negative, so increasing in absolute value) index labels. The keyword arguments `order` and
`output` allow to change these defaults.

The advantage of the macro `@ncon` over the function call `ncon` is that, if `tensorlist` is
not just some variable but an actual list (as a tuple with parentheses or a vector with
square brackets) at the call site, the `@ncon` macro will scan for conjugation calls, e.g.
`conj(A)`, and replace this with just `A` but build a matching list of conjugation flags to
be specified to `ncon`. This makes it more convenient to specify tensor conjugation, without
paying the cost of actively performing the conjugation beforehand.

See also the function [`ncon`](@ref).
"""
macro ncon(args...)
    if length(args) == 2
        return _nconmacro(args[1], args[2])
    else
        return _nconmacro(args[2], args[3], args[1])
    end
end
function _nconmacro(tensors, indices, kwargs = nothing)
    if !(tensors isa Expr) # there is not much that we can do
        if kwargs === nothing
            ex = Expr(
                :call, :ncon, tensors, indices,
                Expr(:call, :fill, false, Expr(:call, :length, tensors))
            )
        else
            ex = Expr(
                :call, :ncon, kwargs, tensors, indices,
                Expr(:call, :fill, false, Expr(:call, :length, tensors))
            )
        end
        return esc(ex)
    end
    if tensors.head == :vect || tensors.head == :tuple
        tensorargs = tensors.args
    elseif tensors.head == :ref
        tensorargs = tensors.args[2:end]
    else
        throw(ArgumentError("invalid @ncon syntax"))
    end
    if any(isa(ta, Expr) && ta.head === :... for ta in tensorargs)
        throw(ArgumentError("@ncon does not support splats (...) in tensor lists."))
    end
    conjlist = fill(false, length(tensorargs))
    for i in 1:length(tensorargs)
        if tensorargs[i] isa Expr
            if tensorargs[i].head == :call && tensorargs[i].args[1] == :conj
                tensorargs[i] = tensorargs[i].args[2]
                conjlist[i] = true
            end
        end
    end
    if tensors.head == :ref
        tensorex = Expr(:ref, tensors.args[1], tensorargs...)
    else
        tensorex = Expr(:ref, :Any, tensorargs...)
    end
    if kwargs === nothing
        ex = Expr(:call, :ncon, tensorex, indices, conjlist)
    else
        ex = Expr(:call, :ncon, kwargs, tensorex, indices, conjlist)
    end
    return esc(ex)
end

"""
    @cutensor tensor_expr

Use the GPU to perform all tensor operations, through the use of the cuTENSOR library. This
will transfer all arrays to the GPU before performing the requested operations. If the
output is an existing host array, the result will be transferred back. If a new array is
created (i.e. using `:=`), it will remain on the GPU device and it is up to the user to
transfer it back. This macro is equivalent to

    @tensor backend = cuTENSORBackend() allocator = CUDAAllocator() tensor_expr

!!! note
    This macro requires the cuTENSOR library to be installed and loaded. This can be
    achieved by running `using cuTENSOR` or `import cuTENSOR` before using the macro.
"""
macro cutensor(ex...)
    if length(methods(_cutensor)) == 0
        throw(ArgumentError("cuTENSOR.jl not loaded: add `using cuTENSOR` or `import cuTENSOR` before using `@cutensor`"))
    end
    return esc(_cutensor(__source__, ex...))
end

function _cutensor end

"""
    @butensor tensor_expr

Use Bumper.jl to handle allocation of temporary tensors. This macro will use the default
buffer and automatically reset it after the tensor expression has been evaluated. This macro
is equivalent to `@no_escape @tensor tensor_expr` with all temporary allocations handled by
Bumper.jl.

!!! note
    This macro requires Bumper.jl to be installed and loaded. This can be achieved by running
    `using Bumper` or `import Bumper` before using the macro.
"""
macro butensor(ex...)
    if length(methods(_butensor)) == 0
        throw(ArgumentError("Bumper.jl not loaded: add `using Bumper` or `import Bumper` before using `@butensor`"))
    end
    return esc(_butensor(__source__, ex...))
end

# construction to move implementation to a package extension
function _butensor end
