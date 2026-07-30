# Verifiers: tools to verify if an expression is a valid tensor expression of certain type

# test for a valid index
const prime = Symbol("'")

"""
    isindex(ex)

Test for a valid index, namely a symbol or integer, or an expression of the form `i′` where `i` is itself a valid index.
"""
function isindex(ex)
    if isa(ex, Symbol) || isa(ex, Int)
        return true
    elseif isexpr(ex, prime) && length(ex.args) == 1
        return isindex(ex.args[1])
    else
        return false
    end
end

# test for a simple tensor object indexed by valid indices
"""
    istensor(ex)

Test for a simple tensor object indexed by valid indices. This means an expression of the form:
    
    A[i, j, k, ...]
    A[i j k ...]
    A[i j k ...; l m ...]
    A[(i, j, k, ...); (l, m, ...)]
    
where `i`, `j`, `k`, ... are valid indices.
"""
function istensor end
istensor(ex) = false
function istensor(ex::Expr)
    if ex.head == :ref || ex.head == :typed_hcat
        if length(ex.args) == 1
            return true
        elseif isexpr(ex.args[2], :parameters)
            return all(isindex, ex.args[2].args) && all(isindex, ex.args[3:end])
        else
            return all(isindex, ex.args[2:end])
        end
    elseif ex.head == :typed_vcat && length(ex.args) == 3
        if isexpr(ex.args[2], [:row, :tuple])
            all(isindex, ex.args[2].args) || return false
        else
            isindex(ex.args[2]) || return false
        end
        if isexpr(ex.args[3], [:row, :tuple])
            all(isindex, ex.args[3].args) || return false
        else
            isindex(ex.args[3]) || return false
        end
        return true
    end
    return false
end

# test for a generalized tensor, i.e. with scalar multiplication and conjugation
isgeneraltensor(ex) = false
function isgeneraltensor(ex::Expr)
    if istensor(ex)
        return true
    elseif ex.head == :call && ex.args[1] == :+ && length(ex.args) == 2
        # unary plus
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :- && length(ex.args) == 2
        # unary minus
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2
        # conjugation
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :*
        # scalar multiplication
        count = 0
        for i in 2:length(ex.args)
            if isgeneraltensor(ex.args[i])
                count += 1
            elseif !isscalarexpr(ex.args[i])
                return false
            end
        end
        return count == 1
    elseif ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3
        # scalar multiplication
        return (isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2]))
    elseif ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3
        # scalar multiplication
        return (isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3]))
        # TODO: disable these operations?
    elseif ex.head == :call && ex.args[1] == :adjoint && length(ex.args) == 2
        # adjoint
        return isgeneraltensor(ex.args[2])
    elseif ex.head == prime && length(ex.args) == 1
        # adjoint
        return isgeneraltensor(ex.args[1])
        # elseif ex.head == :call && ex.args[1] == :transpose && length(ex.args) == 2
        #     # transposition
        #     return isgeneraltensor(ex.args[2])
    end
    return false
end

function hastraceindices(ex)
    obj, leftind, rightind, = decomposegeneraltensor(ex)
    allind = vcat(leftind, rightind)
    return length(allind) != length(unique(allind))
end

# test for a scalar expression, i.e. no indices
"""
    isscalarexpr(ex)
    
Test for a scalar expression, i.e. an expression that can be evaluated to a scalar.
"""
function isscalarexpr(ex)
    if ex isa Symbol || ex isa Number
        return true
    elseif isexpr(ex, :call) && ex.args[1] == :tensorscalar
        return istensorexpr(ex.args[2])
    elseif isexpr(ex, (:ref, :typed_vcat, :typed_hcat))
        return false
    elseif isexpr(ex, :call) # || isdefinition(ex) || isassignment(ex)
        return all(isscalarexpr, ex.args[2:end])
    else
        return true # assume everything else is valid scalar code
    end
end

# test for a tensor contraction expression
function istensorcontraction(ex)
    if isexpr(ex, :call) && ex.args[1] == :*
        return count(istensorexpr, ex.args[2:end]) >= 2
    end
    return false
end

"""
    istensorexpr(ex)

Test for a tensor expression. This means an expression which can be evaluated to a valid
    tensor. This includes:

    A[...] + B[...] - C[...] - ...
    A[...] * B[...] * ...
    λ * A[...] / μ
    λ \\ conj(A[...])
    A[...]' + adjoint(B[...]) - ...
"""
function istensorexpr(ex)
    isgeneraltensor(ex) && return true
    if isexpr(ex, :call)
        if (ex.args[1] == :+ || ex.args[1] == :-)
            return all(istensorexpr, ex.args[2:end]) # all arguments should be tensor expressions (we are not checking matching indices yet)
        elseif ex.args[1] == :*
            count = 0
            for i in 2:length(ex.args)
                if istensorexpr(ex.args[i])
                    count += 1
                elseif !isscalarexpr(ex.args[i])
                    return false
                end
            end
            return count > 0
        elseif ex.args[1] == :/ && length(ex.args) == 3
            return istensorexpr(ex.args[2]) && isscalarexpr(ex.args[3])
        elseif ex.args[1] == :\ && length(ex.args) == 3
            return istensorexpr(ex.args[3]) && isscalarexpr(ex.args[2])
        elseif ex.args[1] == :conj && length(ex.args) == 2
            return istensorexpr(ex.args[2])
        end
    end
    # TODO: disable these?
    if isexpr(ex, :call) && ex.args[1] == :adjoint && length(ex.args) == 2
        return istensorexpr(ex.args[2])
    elseif isexpr(ex, prime)
        return istensorexpr(ex.args[1])
    end
    return false
end

"""
    verifyindices(ex) -> ex

Verify that all index labels in `ex` obey the strict Einstein summation convention, and throw an `ArgumentError` if not.
This convention entails that within a single term, every index label should appear either once (an open index) or exactly twice (a contracted index, either between two different tensors or within a single tensor as a trace).
Parentheses group the factors of a term and thus determine the contraction order, but do not introduce a new scope for the index labels.
Different terms of a sum, different statements, and the argument of an explicit `tensorscalar` call do constitute separate scopes, in which labels can be reused freely.

This routine expects the indices to be normalized, i.e. it should be called after [`normalizeindices`](@ref).
The expression is returned unchanged, such that this can be used as a preprocessor.
"""
function verifyindices(ex)
    if isexpr(ex, :macrocall) && ex.args[1] == Symbol("@notensor")
        return ex
    elseif istensor(ex) || istensorexpr(ex)
        _indexscope(ex)
    elseif isa(ex, Expr)
        foreach(verifyindices, ex.args)
    end
    return ex
end

# analyze a single term and return its open indices, along with the indices that have been
# contracted (closed) within that term; throws if the Einstein convention is violated
function _indexscope(ex)
    if istensor(ex)
        _, leftind, rightind = decomposetensor(ex)
        allind = vcat(leftind, rightind)
        open, closed = Any[], Any[]
        for label in unique(allind)
            n = count(isequal(label), allind)
            if n == 1
                push!(open, label)
            elseif n == 2
                push!(closed, label)
            else
                throw(ArgumentError("@tensor: index $label appears $n times in tensor $ex"))
            end
        end
        return open, closed
    elseif isexpr(ex, :call)
        if ex.args[1] == :tensorscalar
            # separate scope: verify independently and hide the labels from the outer term
            foreach(verifyindices, ex.args[2:end])
            return Any[], Any[]
        elseif (ex.args[1] == :+ || ex.args[1] == :-) && length(ex.args) > 2
            return _indexscope_sum(ex)
        elseif ex.args[1] == :*
            return _indexscope_product(ex)
        elseif ex.args[1] == :/ && length(ex.args) == 3
            verifyindices(ex.args[3])
            return _indexscope(ex.args[2])
        elseif ex.args[1] == :\ && length(ex.args) == 3
            verifyindices(ex.args[2])
            return _indexscope(ex.args[3])
        elseif length(ex.args) == 2 # unary plus or minus, conj, adjoint, ...
            return _indexscope(ex.args[2])
        end
    elseif isexpr(ex, prime) && length(ex.args) == 1
        return _indexscope(ex.args[1])
    end
    return Any[], Any[]
end

function _indexscope_sum(ex)
    open = nothing
    for term in ex.args[2:end]
        if !istensorexpr(term)
            verifyindices(term)
            continue
        end
        openterm, = _indexscope(term)
        if isnothing(open)
            open = openterm
        elseif Set(openterm) != Set(open)
            throw(
                ArgumentError(
                    "@tensor: non-matching indices $(tuple(open...)) and $(tuple(openterm...)) between terms of $ex"
                )
            )
        end
    end
    # the contracted labels of the individual terms are not visible outside of that term
    return @something(open, Any[]), Any[]
end

function _indexscope_product(ex)
    opens, closeds = Any[], Any[]
    for factor in ex.args[2:end]
        if !istensorexpr(factor)
            verifyindices(factor) # scalar factor: cannot contribute index labels
            continue
        end
        openfactor, closedfactor = _indexscope(factor)
        append!(opens, openfactor)
        append!(closeds, closedfactor)
    end
    # labels that are already contracted cannot be reused within this term
    for label in unique(closeds)
        n = 2 * count(isequal(label), closeds) + count(isequal(label), opens)
        n > 2 && throw(ArgumentError("@tensor: index $label appears $n times in $ex"))
    end
    open, closed = Any[], closeds
    for label in unique(opens)
        n = count(isequal(label), opens)
        if n == 1
            push!(open, label)
        elseif n == 2
            push!(closed, label)
        else
            throw(ArgumentError("@tensor: index $label appears $n times in $ex"))
        end
    end
    return open, closed
end

"""
    isassignment(ex)

Test if `ex` is an assignment expression, i.e. `ex` is of one of the forms:

    a = b
    a += b
    a -= b
"""
isassignment(ex) = isexpr(ex, [:(=), :(+=), :(-=)])

"""
    isdefinition(ex)

Test if `ex` is a definition expression, i.e. `ex` is of the form:

    a := b
    a ≔ b
"""
isdefinition(ex) = isexpr(ex, [:(:=), :(≔)])
