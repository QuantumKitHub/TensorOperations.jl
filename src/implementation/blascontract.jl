# general implementation for backends that implement tensor contractions by permuting and
# reshaping the input tensors and then calling a BLAS routine to perform the contraction

# all of the following methods expect that basic argument checks on dimensionality and
# permutation validity have already been performed
function blas_contract!(C, A, pA, B, pB, pAB, α, β, backend, allocator)
    rpA = reverse(pA)
    rpB = reverse(pB)
    # note: `pAB` is only ever consumed through `linearize`/`invperm` from here on, so it is
    # canonicalized to an `IndexTuple` and the reversed permutation does not need to be
    # repartitioned the way `pAB` is
    pAB = linearize(pAB)
    rpAB = let N₁ = numout(pA), N₂ = numin(pB)
        map(n -> ifelse(n > N₁, n - N₁, n + N₂), pAB)
    end
    cp = allocator_checkpoint!(allocator)
    if contract_memcost(C, A, pA, B, pB, pAB) <= contract_memcost(C, B, rpB, A, rpA, rpAB)
        C = _blas_contract!(C, A, pA, B, pB, pAB, α, β, backend, allocator)
    else
        C = _blas_contract!(C, B, rpB, A, rpA, rpAB, α, β, backend, allocator)
    end
    allocator_reset!(allocator, cp)
    return C
end
# specialised fast path for matrix matrix multiplication
function blas_contract!(
        C::StridedView{T, 2},
        A::StridedView{T, 2}, pA::Index2Tuple{1, 1},
        B::StridedView{T, 2}, pB::Index2Tuple{1, 1},
        pAB::IndexTuple{2},
        α::Number, β::Number,
        backend, allocator
    ) where {T}
    A′ = pA == ((1,), (2,)) ? A : transpose(A)
    B′ = pB == ((1,), (2,)) ? B : transpose(B)
    if pAB == (1, 2)
        mul!(C, A′, B′, α, β)
    elseif pAB == (2, 1)
        mul!(C, transpose(B′), transpose(A′), α, β)
    end
    return C
end

# implement necessary permutations - each input is made BLAS-contractable and reshaped into a
# matrix in a separate step within each branch, to avoid boxing/type instabilities from
# returning a union of the input and its permuted copy; as the subsequent steps only receive
# matrices, they no longer depend on the number of indices of the inputs
function _blas_contract!(C, A, pA, B, pB, pAB, α, β, backend, allocator)
    ipAB = oindABinC(pAB, pA, pB)
    if isblascontractable(A, pA) && eltype(A) == eltype(C)
        return _blas_contract_B!(C, _blas_matrix(A, pA), B, pB, pAB, ipAB, α, β, backend, allocator)
    else
        A_ = tensoralloc_add(eltype(C), A, pA, false, Val(true), allocator)
        Anew = tensoradd!(SV(A_), A, pA, false, One(), Zero(), backend, allocator)
        Amat = _blas_matrix(Anew, trivialpermutation(pA))
        _blas_contract_B!(C, Amat, B, pB, pAB, ipAB, α, β, backend, allocator)
        tensorfree!(A_, allocator)
        return C
    end
end
function _blas_contract_B!(C, A, B, pB, pAB, ipAB, α, β, backend, allocator)
    if isblascontractable(B, pB) && eltype(B) == eltype(C)
        return _blas_contract_C!(C, A, _blas_matrix(B, pB), pAB, ipAB, α, β, backend, allocator)
    else
        B_ = tensoralloc_add(eltype(C), B, pB, false, Val(true), allocator)
        Bnew = tensoradd!(SV(B_), B, pB, false, One(), Zero(), backend, allocator)
        Bmat = _blas_matrix(Bnew, trivialpermutation(pB))
        _blas_contract_C!(C, A, Bmat, pAB, ipAB, α, β, backend, allocator)
        tensorfree!(B_, allocator)
        return C
    end
end
function _blas_contract_C!(C, A, B, pAB, ipAB, α, β, backend, allocator)
    if isblasdestination(C, ipAB)
        mul!(_blas_matrix(C, ipAB), A, B, α, β)
    else
        TC = eltype(C)
        C_ = SV(tensoralloc_add(TC, C, ipAB, false, Val(true), allocator))
        mul!(_blas_matrix(C_, trivialpermutation(ipAB)), A, B, one(TC), zero(TC))
        # `C` already exists, so only the permutation of `pAB` matters here and it can be
        # handed over in the canonical `{N,0}` partition
        tensoradd!(C, C_, (pAB, ()), false, α, β, backend, allocator)
        tensorfree!(C_.parent, allocator)
    end
    return C
end

# reshape `A` into the matrix with the indices `p[1]` as rows and `p[2]` as columns, assuming
# this is possible without any further allocations
function _blas_matrix(A::StridedView, p::Index2Tuple)
    sz = size(A)
    m = prod(TupleTools.getindices(sz, p[1]))
    n = prod(TupleTools.getindices(sz, p[2]))
    return sreshape(permutedims(A, linearize(p)), (m, n))
end

function isblascontractable(A::StridedView, p::Index2Tuple)
    eltype(A) <: LinearAlgebra.BlasFloat || return false

    sizeA = size(A)
    stridesA = strides(A)
    sizeA1 = TupleTools.getindices(sizeA, p[1])
    sizeA2 = TupleTools.getindices(sizeA, p[2])
    stridesA1 = TupleTools.getindices(stridesA, p[1])
    stridesA2 = TupleTools.getindices(stridesA, p[2])

    canfuse1, _, s1 = _canfuse(sizeA1, stridesA1)
    canfuse2, _, s2 = _canfuse(sizeA2, stridesA2)

    if A.op == conj
        return canfuse1 && canfuse2 && s2 == 1
    else
        return canfuse1 && canfuse2 && (s1 == 1 || s2 == 1)
    end
end

function isblasdestination(A::StridedView, p::Index2Tuple)
    (eltype(A) <: LinearAlgebra.BlasFloat && A.op == identity) || return false

    sizeA = size(A)
    stridesA = strides(A)

    sizeA1 = TupleTools.getindices(sizeA, p[1])
    stridesA1 = TupleTools.getindices(stridesA, p[1])
    canfuse1, _, s1 = _canfuse(sizeA1, stridesA1)
    (canfuse1 && s1 == 1) || return false

    sizeA2 = TupleTools.getindices(sizeA, p[2])
    stridesA2 = TupleTools.getindices(stridesA, p[2])
    canfuse2, _, _ = _canfuse(sizeA2, stridesA2)
    return canfuse2
end

_canfuse(::Dims{0}, ::Dims{0}) = true, 1, 1
_canfuse(dims::Dims{1}, strides::Dims{1}) = true, dims[1], strides[1]
function _canfuse(dims::Dims{N}, strides::Dims{N}) where {N}
    if dims[1] == 0
        return true, 0, 1
    elseif dims[1] == 1
        return _canfuse(Base.tail(dims), Base.tail(strides))
    else
        b, d, s = _canfuse(Base.tail(dims), Base.tail(strides))
        if b && (s == dims[1] * strides[1] || d == 1)
            dnew = dims[1] * d
            return true, dnew, (dnew == 0 || dnew == 1) ? 1 : strides[1]
        else
            return false, dims[1] * d, strides[1]
        end
    end
end

function oindABinC(pAB, pA, pB)
    ipAB = invperm(linearize(pAB))
    oindAinC = TupleTools.getindices(ipAB, trivialpermutation(pA[1]))
    oindBinC = TupleTools.getindices(ipAB, numout(pA) .+ trivialpermutation(pB[2]))
    return (oindAinC, oindBinC)
end

function contract_memcost(C, A, pA, B, pB, pAB)
    ipAB = oindABinC(pAB, pA, pB)
    return length(A) * (!isblascontractable(A, pA) || eltype(A) !== eltype(C)) +
        length(B) * (!isblascontractable(B, pB) || eltype(B) !== eltype(C)) +
        length(C) * !isblasdestination(C, ipAB)
end
