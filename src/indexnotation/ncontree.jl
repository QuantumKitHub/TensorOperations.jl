# Verify if a list of indices specifies a tensor contraction in ncon style.
check_nconstyle(::Type{Bool}, network) = _check_nconstyle(network, Val(true))
check_nconstyle(network) = _check_nconstyle(network, Val(false))

function _check_nconstyle(network, ::Val{check}) where {check}
    allindices = Vector{Int}()
    for ind in network
        for i in ind
            isa(i, Integer) || return check ? false :
                throw(IndexError("All indices must be integers"))
            push!(allindices, i)
        end
    end
    sort!(allindices)
    j = 1
    while j <= length(allindices)
        i = allindices[j]
        # number of times that `i` appears
        n = 1
        while j + n <= length(allindices) && allindices[j + n] == i
            n += 1
        end
        if i > 0 # positive labels represent contractions or traces and should appear twice
            n == 1 && return check ? false :
                throw(IndexError(lazy"Index $i appears only once in the network"))
            n > 2 && return check ? false :
                throw(IndexError(lazy"Index $i appears more than twice in the network"))
        elseif i < 0 # negative labels represent open indices and should appear once
            n > 1 && return check ? false :
                throw(IndexError(lazy"Index $i appears more than once in the network"))
        else # i == 0
            return check ? false : throw(IndexError("Index 0 is not allowed in the network"))
        end
        j += n
    end
    return check ? true : nothing
end

ncontree(network) = _nestedtree(_ncontree_steps(network), length(network))

# Determine the contraction order where the indices are contracted in increasing order, as a
# list of pairwise contractions (see `_mergetrees!`).
function _ncontree_steps(network)
    # collect the positive labels, i.e. the contraction indices, and the tensors they belong to
    contractionindices = Int[]
    owners = Int[]
    for (k, indices) in enumerate(network)
        for i in indices
            i > 0 || continue
            push!(contractionindices, i)
            push!(owners, k)
        end
    end

    # contract the indices in increasing order
    steps = NTuple{2, Int}[]
    partialtrees = collect(1:length(network))
    group = collect(1:length(network))
    for j in sortperm(contractionindices)
        k = findnext(==(contractionindices[j]), contractionindices, j + 1)
        # each index appears twice, only consider the first occurrence
        isnothing(k) || _mergetrees!(steps, partialtrees, group, owners[j], owners[k])
    end
    return _finalizetree!(steps, partialtrees, group)
end

# A contraction tree for a network of `n` tensors is represented as a list of pairwise
# contractions `steps`, where `steps[s] = (i, j)` contracts the (partial) results `i` and `j`.
# Here, results `1:n` are the tensors themselves, and result `n + s` is the result of `steps[s]`.
# The partial tree containing tensor `k` is `partialtrees[group[k]]`.

# Merge the partial trees containing tensors `k1` and `k2`. The merged tree replaces the one
# containing the tensor that comes first, and nothing happens if both tensors are already in
# the same partial tree.
function _mergetrees!(
        steps::Vector{NTuple{2, Int}}, partialtrees::Vector{Int}, group::Vector{Int},
        k1::Int, k2::Int
    )
    g1, g2 = minmax(group[k1], group[k2])
    g1 == g2 && return steps
    push!(steps, (partialtrees[g1], partialtrees[g2]))
    partialtrees[g1] = length(group) + length(steps)
    replace!(group, g2 => g1)
    return steps
end

# Combine the remaining partial trees, which are disconnected and thus require outer products,
# starting from the end.
function _finalizetree!(
        steps::Vector{NTuple{2, Int}}, partialtrees::Vector{Int}, group::Vector{Int}
    )
    n = length(group)
    remaining = unique!(group)
    tree = partialtrees[remaining[end]]
    for g in Iterators.drop(Iterators.reverse(remaining), 1)
        push!(steps, (partialtrees[g], tree))
        tree = n + length(steps)
    end
    return steps
end

# convert a list of pairwise contractions to a nested contraction tree
function _nestedtree(steps::Vector{NTuple{2, Int}}, n::Int)
    trees = collect(Any, 1:n)
    for (i, j) in steps
        push!(trees, Any[trees[i], trees[j]])
    end
    return trees[end]
end

"""
    nconindexcompletion(ex)

Complete the indices of the left hand side of an ncon expression. For example, the following expressions are equivalent after index completion.

    @tensor A[:] := B[-1, 1, 2] * C[1, 2, -3]
    @tensor A[-1, -2] := B[-1, 1, 2] * C[1, 2, -3]
"""
function nconindexcompletion(ex)
    if isassignment(ex) || isdefinition(ex)
        lhs, rhs = getlhs(ex), getrhs(ex)
        # process left hand side
        if istensor(lhs) && istensorexpr(rhs)
            indices = getindices(rhs)

            if lhs.head == :ref && length(lhs.args) == 2 && lhs.args[2] == :(:)
                if all(isa(i, Integer) && i < 0 for i in indices)
                    lhs = Expr(:ref, lhs.args[1], sort(indices; rev = true)...)
                else
                    error("cannot automatically infer index order of left hand side")
                end
            end
            return Expr(ex.head, lhs, rhs)
        else
            return ex
        end
    elseif ex isa Expr
        return Expr(ex.head, map(nconindexcompletion, ex.args)...)
    else
        return ex
    end
end

function resolve_traces(tensors, network)
    transformed = map(zip(tensors, network)) do (A, IA)
        IC = unique2(IA)
        return length(IC) == length(IA) ? (A, IA) : (tensortrace(IC, A, IA), IC)
    end
    return first.(transformed), last.(transformed)
end
