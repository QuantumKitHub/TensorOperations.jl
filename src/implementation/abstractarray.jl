# ------------------------------------------------------------------------------------------
# General definitions for AbstractArray instances
# ------------------------------------------------------------------------------------------
tensorscalar(C::AbstractArray) = ndims(C) == 0 ? sum(C) : throw(DimensionMismatch())
# this function more or less boils down to `fill!(similar(x), y)` but does so in a single
# call to allow higher-order derivatives
function similar_and_fill(x, y)
    x′ = tensoralloc(typeof(x), tensorstructure(x))
    return fill!(x′, y)
end
# sum is a trick to get the scalar value of a 0-dimensional array, that also works on CuArray

tensorcost(C::AbstractArray, i) = size(C, i)

function checkcontractible(
        A::AbstractArray, iA, conjA::Bool,
        B::AbstractArray, iB, conjB::Bool, label
    )
    size(A, iA) == size(B, iB) ||
        throw(DimensionMismatch("Nonmatching dimensions for $label: $(size(A, iA)) != $(size(B, iB))"))
    return nothing
end

# ------------------------------------------------------------------------------------------
# Default backend selection mechanism for AbstractArray instances
# ------------------------------------------------------------------------------------------
function select_backend(::typeof(tensoradd!), C::AbstractArray, A::AbstractArray)
    if isstrided(A) && isstrided(C)
        return select_backend(tensoradd!, StridedView(C), StridedView(A))
    else
        return BaseView()
    end
end

function select_backend(::typeof(tensortrace!), C::AbstractArray, A::AbstractArray)
    if isstrided(A) && isstrided(C)
        return select_backend(tensortrace!, StridedView(C), StridedView(A))
    else
        return BaseView()
    end
end

function select_backend(
        ::typeof(tensorcontract!), C::AbstractArray, A::AbstractArray, B::AbstractArray
    )
    if all(_isstridedordiag, (A, B, C))
        return select_backend(
            tensorcontract!, _stridedordiag(C), _stridedordiag(A),
            _stridedordiag(B)
        )
    else
        if eltype(C) <: LinearAlgebra.BlasFloat
            return BaseCopy()
        else
            return BaseView()
        end
    end
end
_isstridedordiag(A::AbstractArray) = isstrided(A) || isa(A, Diagonal)
_stridedordiag(A::AbstractArray) = StridedView(A)
_stridedordiag(A::Diagonal) = A

# ------------------------------------------------------------------------------------------
# Argument Checking: can be used by backends to check the validity of the arguments
# ------------------------------------------------------------------------------------------

"""
    argcheck_index2tuple(C::AbstractArray, pC::Index2Tuple)

Check that `C` has `numind(pC)` indices and that `pC` constitutes a valid permutation.
"""
function argcheck_index2tuple(C::AbstractArray, pC::Index2Tuple)
    return ndims(C) == numind(pC) && isperm(linearize(pC)) ||
        throw(IndexError("invalid permutation of length $(ndims(C)): $pC"))
end

"""
    argcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple)

Check that `C` and `A` have `numind(pA)` indices and that `pA` constitutes a valid permutation.
"""
function argcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple)
    ndims(C) == ndims(A) || throw(IndexError("non-matching number of dimensions"))
    argcheck_index2tuple(A, pA)
    return nothing
end

"""
    argcheck_tensortrace(C::AbstractArray, A::AbstractArray, p::Index2Tuple, q::Index2Tuple)

Check that the partial trace of `A` over indices `q` and with permutation of the remaining
indices `p` is compatible with output `C`.
"""
function argcheck_tensortrace(
        C::AbstractArray, A::AbstractArray, p::Index2Tuple, q::Index2Tuple
    )
    ndims(C) == numind(p) ||
        throw(IndexError("invalid selection of length $(ndims(C)): $p"))
    2 * numin(q) == 2 * numout(q) == ndims(A) - ndims(C) ||
        throw(IndexError("invalid number of trace dimensions"))
    argcheck_index2tuple(A, ((p[1]..., q[1]...), (p[2]..., q[2]...)))
    return nothing
end

"""
    argcheck_tensorcontract(C::AbstractArray, A::AbstractArray, pA::Index2Tuple, B::AbstractArray, pB::Index2Tuple, pAB::Index2Tuple)

Check that `C`, `A` and `pA`, and `B` and `pB` and `pAB` have compatible indices and number
of dimensions.
"""
function argcheck_tensorcontract(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple,
        B::AbstractArray, pB::Index2Tuple,
        pAB::Index2Tuple
    )
    argcheck_index2tuple(C, pAB)
    argcheck_index2tuple(A, pA)
    argcheck_index2tuple(B, pB)
    numout(pA) + numin(pB) == ndims(C) ||
        throw(IndexError("non-matching output indices in contraction"))
    numin(pA) == numout(pB) ||
        throw(IndexError("non-matching input indices in contraction"))
    return nothing
end

"""
    dimcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple)

Check that `C` and `A` have compatible sizes for the addition specified by `pA`.
"""
function dimcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple)
    szA, szC = size(A), size(C)
    TupleTools.getindices(szA, linearize(pA)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end

"""
    dimcheck_tensorcontract(C::AbstractArray, A::AbstractArray,
                            p::Index2Tuple, q::Index2Tuple)

Check that `C` and `A` have compatible sizes for the trace and addition specified by `p` and
`q`.
"""
function dimcheck_tensortrace(
        C::AbstractArray, A::AbstractArray, p::Index2Tuple, q::Index2Tuple
    )
    szA, szC = size(A), size(C)
    TupleTools.getindices(szA, q[1]) == TupleTools.getindices(szA, q[2]) ||
        throw(DimensionMismatch("non-matching sizes in traced dimensions"))
    TupleTools.getindices(szA, linearize(p)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end

"""
    dimcheck_tensorcontract(C::AbstractArray,
                            A::AbstractArray, pA::Index2Tuple,
                            B::AbstractArray, pB::Index2Tuple,
                            pAB::Index2Tuple)

Check that `C`, `A` and `B` have compatible sizes for the contraction specified by `pA`,
`pB` and `pAB`.
"""
function dimcheck_tensorcontract(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple,
        B::AbstractArray, pB::Index2Tuple,
        pAB::Index2Tuple
    )
    szA, szB, szC = size(A), size(B), size(C)
    TupleTools.getindices(szA, pA[2]) == TupleTools.getindices(szB, pB[1]) ||
        throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    szAB = (TupleTools.getindices(szA, pA[1])..., TupleTools.getindices(szB, pB[2])...)
    TupleTools.getindices(szAB, linearize(pAB)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end
