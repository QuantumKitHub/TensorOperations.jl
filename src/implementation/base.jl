#-------------------------------------------------------------------------------------------
# AbstractArray implementation based on Base + LinearAlgebra
#-------------------------------------------------------------------------------------------
# Note that this is mostly for convenience + checking, and not for performance
function tensoradd!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        ::BaseView, allocator = DefaultAllocator()
    )
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)

    # can we assume that C is mutable?
    # is there more functionality in base that we can use?
    if conjA
        if iszero(β)
            C .= α .* conj.(PermutedDimsArray(A, linearize(pA)))
        else
            C .= β .* C .+ α .* conj.(PermutedDimsArray(A, linearize(pA)))
        end
    else
        if iszero(β)
            C .= α .* PermutedDimsArray(A, linearize(pA))
        else
            C .= β .* C .+ α .* PermutedDimsArray(A, linearize(pA))
        end
    end
    return C
end

# needed to avoid ambiguity error
function tensoradd!(
        C::AbstractArray,
        A::Diagonal, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        ::BaseView, allocator = DefaultAllocator()
    )
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)

    # can we assume that C is mutable?
    # is there more functionality in base that we can use?
    if conjA
        if iszero(β)
            C .= α .* conj.(A)
        else
            C .= β .* C .+ α .* conj.(A)
        end
    else
        if iszero(β)
            C .= α .* A
        else
            C .= β .* C .+ α .* A
        end
    end
    return C
end
function tensoradd!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        ::BaseCopy, allocator = DefaultAllocator()
    )
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)

    # can we assume that C is mutable?
    # is there more functionality in base that we can use?
    Ã = if !isa(A, Diagonal)
        Atemp = tensoralloc_add(eltype(A), A, pA, conjA, Val(true), allocator)
        permutedims!(Atemp, A, linearize(pA))
    else
        Atemp = nothing
        A
    end
    if conjA
        if iszero(β)
            C .= α .* conj.(Ã)
        else
            C .= β .* C .+ α .* conj.(Ã)
        end
    else
        if iszero(β)
            C .= α .* Ã
        else
            C .= β .* C .+ α .* Ã
        end
    end
    !isa(A, Diagonal) && tensorfree!(Atemp, allocator)
    return C
end

# tensortrace
function tensortrace!(
        C::AbstractArray,
        A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        ::BaseView, allocator = DefaultAllocator()
    )
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)

    szA = size(A)
    so = TupleTools.getindices(szA, linearize(p))
    st = prod(TupleTools.getindices(szA, q[1]))
    Ã = reshape(
        PermutedDimsArray(A, (linearize(p)..., linearize(q)...)), (prod(so), st, st)
    )

    if conjA
        if iszero(β)
            C .= α .* conj.(reshape(view(Ã, :, 1, 1), so))
        else
            C .= β .* C .+ α .* conj.(reshape(view(Ã, :, 1, 1), so))
        end
        for i in 2:st
            C .+= α .* conj.(reshape(view(Ã, :, i, i), so))
        end
    else
        if iszero(β)
            C .= α .* reshape(view(Ã, :, 1, 1), so)
        else
            C .= β .* C .+ α .* reshape(view(Ã, :, 1, 1), so)
        end
        for i in 2:st
            C .+= α .* reshape(view(Ã, :, i, i), so)
        end
    end
    return C
end
function tensortrace!(
        C::AbstractArray,
        A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        ::BaseCopy, allocator = DefaultAllocator()
    )
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)

    szA = size(A)
    so = TupleTools.getindices(szA, linearize(p))
    st = prod(TupleTools.getindices(szA, q[1]))
    perm = (linearize(p)..., linearize(q)...)
    Atemp = tensoralloc_add(eltype(A), A, (perm, ()), conjA, Val(true), allocator)
    Ã = reshape(permutedims!(Atemp, A, perm), (prod(so), st, st))
    if conjA
        if iszero(β)
            C .= α .* conj.(reshape(view(Ã, :, 1, 1), so))
        else
            C .= β .* C .+ α .* conj.(reshape(view(Ã, :, 1, 1), so))
        end
        for i in 2:st
            C .+= α .* conj.(reshape(view(Ã, :, i, i), so))
        end
    else
        if iszero(β)
            C .= α .* reshape(view(Ã, :, 1, 1), so)
        else
            C .= β .* C .+ α .* reshape(view(Ã, :, 1, 1), so)
        end
        for i in 2:st
            C .+= α .* reshape(view(Ã, :, i, i), so)
        end
    end
    tensorfree!(Atemp, allocator)
    return C
end

function tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        ::BaseView, allocator = DefaultAllocator()
    )
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    szA = size(A)
    szB = size(B)
    soA = TupleTools.getindices(szA, pA[1])
    soB = TupleTools.getindices(szB, pB[2])
    sc = TupleTools.getindices(szA, pA[2])
    soA1 = prod(soA)
    soB1 = prod(soB)
    sc1 = prod(sc)
    pC = invperm(linearize(pAB))
    C̃ = reshape(PermutedDimsArray(C, pC), (soA1, soB1))

    if conjA && conjB
        Ã = reshape(PermutedDimsArray(A, linearize(reverse(pA))), (sc1, soA1))
        B̃ = reshape(PermutedDimsArray(B, linearize(reverse(pB))), (soB1, sc1))
        C̃ = mul!(C̃, adjoint(Ã), adjoint(B̃), α, β)
    elseif conjA
        Ã = reshape(PermutedDimsArray(A, linearize(reverse(pA))), (sc1, soA1))
        B̃ = reshape(PermutedDimsArray(B, linearize(pB)), (sc1, soB1))
        C̃ = mul!(C̃, adjoint(Ã), B̃, α, β)
    elseif conjB
        Ã = reshape(PermutedDimsArray(A, linearize(pA)), (soA1, sc1))
        B̃ = reshape(PermutedDimsArray(B, linearize(reverse(pB))), (soB1, sc1))
        C̃ = mul!(C̃, Ã, adjoint(B̃), α, β)
    else
        Ã = reshape(PermutedDimsArray(A, linearize(pA)), (soA1, sc1))
        B̃ = reshape(PermutedDimsArray(B, linearize(pB)), (sc1, soB1))
        C̃ = mul!(C̃, Ã, B̃, α, β)
    end
    return C
end
function tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        ::BaseCopy, allocator = DefaultAllocator()
    )
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    szA = size(A)
    szB = size(B)
    soA = TupleTools.getindices(szA, pA[1])
    soB = TupleTools.getindices(szB, pB[2])
    sc = TupleTools.getindices(szA, pA[2])
    soA1 = prod(soA)
    soB1 = prod(soB)
    sc1 = prod(sc)

    AB = tensoralloc_contract(
        eltype(C), A, pA, conjA, B, pB, conjB,
        trivialpermutation(pAB), Val(true), allocator
    )
    ÃB̃ = reshape(AB, (soA1, soB1))
    if conjA && conjB
        Atemp = tensoralloc_add(eltype(C), A, reverse(pA), conjA, Val(true), allocator)
        Btemp = tensoralloc_add(eltype(C), B, reverse(pB), conjB, Val(true), allocator)
        Ã = adjoint(reshape(permutedims!(Atemp, A, linearize(reverse(pA))), (sc1, soA1)))
        B̃ = adjoint(reshape(permutedims!(Btemp, B, linearize(reverse(pB))), (soB1, sc1)))
    elseif conjA
        Atemp = tensoralloc_add(eltype(C), A, reverse(pA), conjA, Val(true), allocator)
        Btemp = tensoralloc_add(eltype(C), B, pB, conjB, Val(true), allocator)
        Ã = adjoint(reshape(permutedims!(Atemp, A, linearize(reverse(pA))), (sc1, soA1)))
        B̃ = reshape(permutedims!(Btemp, B, linearize(pB)), (sc1, soB1))
    elseif conjB
        Atemp = tensoralloc_add(eltype(C), A, pA, conjA, Val(true), allocator)
        Btemp = tensoralloc_add(eltype(C), B, reverse(pB), conjB, Val(true), allocator)
        Ã = reshape(permutedims!(Atemp, A, linearize(pA)), (soA1, sc1))
        B̃ = adjoint(reshape(permutedims!(Btemp, B, linearize(reverse(pB))), (soB1, sc1)))
    else
        Atemp = tensoralloc_add(eltype(C), A, pA, conjA, Val(true), allocator)
        Btemp = tensoralloc_add(eltype(C), B, pB, conjB, Val(true), allocator)
        Ã = reshape(permutedims!(Atemp, A, linearize(pA)), (soA1, sc1))
        B̃ = reshape(permutedims!(Btemp, B, linearize(pB)), (sc1, soB1))
    end
    mul!(ÃB̃, Ã, B̃)
    tensorfree!(Btemp, allocator)
    tensorfree!(Atemp, allocator)

    if istrivialpermutation(linearize(pAB))
        pAB = AB
    else
        pABtemp = tensoralloc_add(eltype(C), AB, pAB, false, Val(true), allocator)
        pAB = permutedims!(pABtemp, AB, linearize(pAB))
    end
    if iszero(β)
        C .= α .* pAB
    else
        C .= β .* C .+ α .* pAB
    end
    pAB === AB || tensorfree!(pABtemp, allocator)
    tensorfree!(AB, allocator)
    return C
end
