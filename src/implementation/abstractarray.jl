# ------------------------------------------------------------------------------------------
# General definitions for AbstractArray instances
# ------------------------------------------------------------------------------------------
tensorscalar(C::AbstractArray) = ndims(C) == 0 ? sum(C) : throw(DimensionMismatch())
# sum is a trick to get the scalar value of a 0-dimensional array, that also works on CuArray

tensorcost(C::AbstractArray, i) = size(C, i)

function checkcontractible(
        A::AbstractArray, iA, conjA::Bool,
        B::AbstractArray, iB, conjB::Bool, label
    )
    size(A, iA) == size(B, iB) ||
        throw(DimensionMismatch(lazy"Nonmatching dimensions for $label: $(size(A, iA)) != $(size(B, iB))"))
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
argcheck_index2tuple(C::AbstractArray, pC::Index2Tuple) = argcheck_indextuple(C, linearize(pC))
function argcheck_indextuple(C::AbstractArray, pC::IndexTuple)
    ndims(C) == numind(pC) && isperm(pC) ||
        throw(IndexError(lazy"invalid permutation of length $(ndims(C)): $pC"))
    return nothing
end

"""
    argcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple)

Check that `C` and `A` have `numind(pA)` indices and that `pA` constitutes a valid permutation.
"""
argcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple) =
    argcheck_tensoradd(C, A, linearize(pA))
function argcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::IndexTuple)
    ndims(C) == ndims(A) || throw(IndexError("non-matching number of dimensions"))
    argcheck_indextuple(A, pA)
    return nothing
end

"""
    argcheck_tensortrace(C::AbstractArray, A::AbstractArray, p::Index2Tuple, q::Index2Tuple)

Check that the partial trace of `A` over indices `q` and with permutation of the remaining
indices `p` is compatible with output `C`.
"""
argcheck_tensortrace(C::AbstractArray, A::AbstractArray, p::Index2Tuple, q::Index2Tuple) =
    argcheck_tensortrace(C, A, linearize(p), q)
function argcheck_tensortrace(
        C::AbstractArray, A::AbstractArray, p::IndexTuple, q::Index2Tuple
    )
    ndims(C) == numind(p) ||
        throw(IndexError(lazy"invalid selection of length $(ndims(C)): $p"))
    2 * numin(q) == 2 * numout(q) == ndims(A) - ndims(C) ||
        throw(IndexError("invalid number of trace dimensions"))
    argcheck_indextuple(A, (p..., q[1]..., q[2]...))
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
    return argcheck_tensorcontract(C, A, pA, B, pB, linearize(pAB))
end
function argcheck_tensorcontract(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple,
        B::AbstractArray, pB::Index2Tuple,
        pAB::IndexTuple
    )
    argcheck_indextuple(C, pAB)
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
dimcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple) = dimcheck_tensoradd(C, A, linearize(pA))
function dimcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::IndexTuple)
    szA, szC = size(A), size(C)
    TupleTools.getindices(szA, pA) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end

"""
    dimcheck_tensorcontract(C::AbstractArray, A::AbstractArray, p::Index2Tuple, q::Index2Tuple)

Check that `C` and `A` have compatible sizes for the trace and addition specified by `p` and `q`.
"""
dimcheck_tensortrace(C::AbstractArray, A::AbstractArray, p::Index2Tuple, q::Index2Tuple) =
    dimcheck_tensortrace(C, A, linearize(p), q)
function dimcheck_tensortrace(
        C::AbstractArray, A::AbstractArray, p::IndexTuple, q::Index2Tuple
    )
    szA, szC = size(A), size(C)
    TupleTools.getindices(szA, q[1]) == TupleTools.getindices(szA, q[2]) ||
        throw(DimensionMismatch("non-matching sizes in traced dimensions"))
    TupleTools.getindices(szA, p) == szC ||
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
    return dimcheck_tensorcontract(C, A, pA, B, pB, linearize(pAB))
end
function dimcheck_tensorcontract(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple,
        B::AbstractArray, pB::Index2Tuple,
        pAB::IndexTuple
    )
    szA, szB, szC = size(A), size(B), size(C)
    TupleTools.getindices(szA, pA[2]) == TupleTools.getindices(szB, pB[1]) ||
        throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    szAB = (TupleTools.getindices(szA, pA[1])..., TupleTools.getindices(szB, pB[2])...)
    TupleTools.getindices(szAB, pAB) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end
