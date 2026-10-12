"""
    ncon(tensorlist, indexlist, [conjlist, sym]; order = ..., output = ..., backend = ..., allocator = ...)

Contract the tensors in `tensorlist` (of type `Vector` or `Tuple`) according to the network
as specified by `indexlist`. Here, `indexlist` is a list (i.e. a `Vector` or `Tuple`) with
the same length as `tensorlist` whose entries are themselves lists (preferably
`Vector{Int}`) where every integer entry provides a label for corresponding index/dimension
of the corresponding tensor in `tensorlist`. Positive integers are used to label indices
that need to be contracted, and such thus appear in two different entries within
`indexlist`, whereas negative integers are used to label indices of the output tensor, and
should appear only once.

Optional arguments in another list with the same length, `conjlist`, whose entries are of
type `Bool` and indicate whether the corresponding tensor object should be conjugated
(`true`) or not (`false`). The default is `false` for all entries.

By default, contractions are performed in the order such that the indices being contracted
over are labelled by increasing integers, i.e. first the contraction corresponding to label
`1` is performed. The output tensor had an index order corresponding to decreasing
(negative, so increasing in absolute value) index labels. The keyword arguments `order` and
`output` allow to change these defaults.

See also the macro version [`@ncon`](@ref).
"""
function ncon(
        tensors, network, conjlist = fill(false, length(tensors));
        order = nothing, output = nothing, kwargs...
    )
    length(tensors) == length(network) == length(conjlist) ||
        throw(ArgumentError("number of tensors and of index lists should be the same"))
    check_nconstyle(network)
    output′ = nconoutput(network, output)

    if length(tensors) == 1
        if length(output′) == length(network[1])
            return tensorcopy(output′, tensors[1], network[1], conjlist[1]; kwargs...)
        else
            return tensortrace(output′, tensors[1], network[1], conjlist[1]; kwargs...)
        end
    end

    tensors′, network′ = resolve_traces(tensors, network)
    # extract the keyword arguments once, rather than for every pairwise contraction
    backend, allocator = _ncon_kwargs(; kwargs...)
    conjlist′ = _ncon_conjlist(conjlist)
    return _ncon(tensors′, network′, conjlist′, output′, order, backend, allocator)
end

function _ncon(tensors, network, conjlist, output, order, backend, allocator)
    steps = order === nothing ? _ncontree_steps(network) :
        _indexordertree_steps(network, order)
    cp = allocator_checkpoint!(allocator)
    n = length(tensors)
    i, j = steps[end]
    A, IA, conjA = contracttree(tensors, network, conjlist, steps, i, backend, allocator)
    B, IB, conjB = contracttree(tensors, network, conjlist, steps, j, backend, allocator)
    C = _ncon_contract(A, IA, conjA, B, IB, conjB, output, Val(false), backend, allocator)
    i > n && tensorfree!(A, allocator)
    j > n && tensorfree!(B, allocator)
    allocator_reset!(allocator, cp)
    return isempty(output) ? tensorscalar(C) : C
end

# Compute the (partial) result `k` of the contraction tree `steps` (see `_mergetrees!`).
# The types and number of indices of the intermediate tensors are only known at runtime, but
# the labels are kept as `Vector{Int}` such that all index computations are type stable, and
# every pairwise contraction goes through a single dynamic dispatch, in `_ncon_contract`.
function contracttree(tensors, network, conjlist, steps, k::Int, backend, allocator)
    n = length(tensors)
    k <= n && return tensors[k], network[k], conjlist[k]
    i, j = steps[k - n]
    A, IA, conjA = contracttree(tensors, network, conjlist, steps, i, backend, allocator)
    B, IB, conjB = contracttree(tensors, network, conjlist, steps, j, backend, allocator)
    IC = _ncon_symdiff(IA, IB)
    C = _ncon_contract(A, IA, conjA, B, IB, conjB, IC, Val(true), backend, allocator)
    i > n && tensorfree!(A, allocator)
    j > n && tensorfree!(B, allocator)
    return C, IC, false
end

function _ncon_kwargs(; kwargs...)
    args = _kwargs2args(; kwargs...)
    return length(args) == 2 ? args :
        length(args) == 1 ? (args[1], DefaultAllocator()) :
        (DefaultBackend(), DefaultAllocator())
end

# labels of the result of contracting tensors with labels `IA` and `IB`, i.e. all labels that
# appear in only one of them, in order of appearance
# note: this is much faster than `Base.symdiff`, which constructs a `Set`
function _ncon_symdiff(IA::Vector{Int}, IB::Vector{Int})
    IC = Int[]
    for i in IA
        i in IB || push!(IC, i)
    end
    for i in IB
        i in IA || push!(IC, i)
    end
    return IC
end

function _ncon_contract(
        A, IA::Vector{Int}, conjA::Bool, B, IB::Vector{Int}, conjB::Bool, IC::Vector{Int},
        istemp::Val, backend, allocator
    )
    # function barrier: the number of indices of `A`, `B` and `C` is only known at runtime
    IA′, IB′, IC′ = _ncon_tuple(IA), _ncon_tuple(IB), _ncon_tuple(IC)
    return _ncon_contract!(A, IA′, conjA, B, IB′, conjB, IC′, istemp, backend, allocator)
end
function _ncon_contract!(A, IA, conjA, B, IB, conjB, IC, istemp::Val, backend, allocator)
    pA, pB, pAB = contract_indices(IA, IB, IC)
    if istemp === Val(true)
        # intermediates list the open indices of `A` before those of `B`, so partition them as
        # `(open A; open B)`, which avoids repartitioning copies for e.g. symmetric tensors
        pAB = trivialpermutation(numout(pA), numin(pB))
    end
    TC = promote_contract(scalartype(A), scalartype(B))
    C = tensoralloc_contract(TC, A, pA, conjA, B, pB, conjB, pAB, istemp, allocator)
    return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, One(), Zero(), backend, allocator)
end

# convert a list of labels to a tuple
# note: `ntuple` is much faster than `Tuple(I)` or `(I...,)` for short lists
_ncon_tuple(I::Vector{Int}) = ntuple(k -> I[k], length(I))

function nconoutput(network, output)
    outputindices = Vector{Int}()
    for n in network
        for k in n
            if k < 0
                push!(outputindices, k)
            end
        end
    end
    isnothing(output) && return sort(outputindices; rev = true)

    issetequal(output, outputindices) ||
        throw(ArgumentError("invalid NCON network: $network -> $output"))
    return collect(Int, output)
end

_ncon_conjlist(conjlist) = collect(Bool, conjlist)
