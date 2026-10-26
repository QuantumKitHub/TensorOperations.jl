# verify that every contraction index appears exactly twice, and any other index exactly once
function _verify_indexorder(network, indexorder)
    for indices in network
        for i in indices
            n = sum(ind -> count(isequal(i), ind), network)
            n == (i in indexorder ? 2 : 1) || return false
        end
    end
    return true
end

function indexordertree(network, indexorder)
    return _nestedtree(_indexordertree_steps(network, indexorder), length(network))
end

# Determine the contraction order where the indices are contracted in the order specified by
# `indexorder`, as a list of pairwise contractions (see `_mergetrees!`).
function _indexordertree_steps(network, indexorder)
    _verify_indexorder(network, indexorder) ||
        throw(ArgumentError("not a valid contraction tree with contraction indices $indexorder"))
    steps = NTuple{2, Int}[]
    partialtrees = collect(1:length(network))
    group = collect(1:length(network))
    for c in indexorder
        # trace indices have already been removed from the network
        k1 = findfirst(∋(c), network)
        isnothing(k1) && continue
        k2 = findnext(∋(c), network, k1 + 1)
        @assert !isnothing(k2) # guaranteed by `_verify_indexorder`
        _mergetrees!(steps, partialtrees, group, k1, k2)
    end
    return _finalizetree!(steps, partialtrees, group)
end
