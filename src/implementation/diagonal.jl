#-------------------------------------------------------------------------------------------
# Specialized implementations for contractions involving diagonal matrices
#-------------------------------------------------------------------------------------------
function tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::Diagonal, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        ::StridedNative, allocator = DefaultAllocator()
    )
    @nospecialize allocator

    # standardize input types for compilation time
    α′ = standardize_scalartype(C, α)
    β′ = standardize_scalartype(C, β)
    pAB′ = linearize(pAB)

    argcheck_tensorcontract(C, A, pA, B, pB, pAB′)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB′)

    if conjA && conjB
        _diagtensorcontract!(SV(C), conj(SV(A)), pA, conj(SV(B.diag)), pB, pAB′, α′, β′)
    elseif conjA
        _diagtensorcontract!(SV(C), conj(SV(A)), pA, SV(B.diag), pB, pAB′, α′, β′)
    elseif conjB
        _diagtensorcontract!(SV(C), SV(A), pA, conj(SV(B.diag)), pB, pAB′, α′, β′)
    else
        _diagtensorcontract!(SV(C), SV(A), pA, SV(B.diag), pB, pAB′, α′, β′)
    end
    return C
end

function tensorcontract!(
        C::AbstractArray,
        A::Diagonal, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        ::StridedNative, allocator = DefaultAllocator()
    )
    @nospecialize allocator

    # standardize input types for compilation time
    α′ = standardize_scalartype(C, α)
    β′ = standardize_scalartype(C, β)
    pAB′ = linearize(pAB)

    argcheck_tensorcontract(C, A, pA, B, pB, pAB′)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB′)

    rpA = reverse(pA)
    rpB = reverse(pB)
    # note: `pAB` is only ever consumed through `linearize`/`invperm`, so the reversed
    # permutation does not need to be repartitioned as `pAB` is
    rpAB = let N₁ = numout(pA), N₂ = numin(pB)
        map(n -> ifelse(n > N₁, n - N₁, n + N₂), pAB′)
    end

    if conjA && conjB
        _diagtensorcontract!(SV(C), conj(SV(B)), rpB, conj(SV(A.diag)), rpA, rpAB, α′, β′)
    elseif conjA
        _diagtensorcontract!(SV(C), SV(B), rpB, conj(SV(A.diag)), rpA, rpAB, α′, β′)
    elseif conjB
        _diagtensorcontract!(SV(C), conj(SV(B)), rpB, SV(A.diag), rpA, rpAB, α′, β′)
    else
        _diagtensorcontract!(SV(C), SV(B), rpB, SV(A.diag), rpA, rpAB, α′, β′)
    end
    return C
end

function tensorcontract!(
        C::AbstractArray,
        A::Diagonal, pA::Index2Tuple, conjA::Bool,
        B::Diagonal, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        ::StridedNative, allocator = DefaultAllocator()
    )
    @nospecialize allocator

    # standardize input types for compilation time
    α′ = standardize_scalartype(C, α)
    β′ = standardize_scalartype(C, β)
    pAB′ = linearize(pAB)

    argcheck_tensorcontract(C, A, pA, B, pB, pAB′)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB′)

    if conjA && conjB
        _diagdiagcontract!(SV(C), conj(SV(A.diag)), pA, conj(SV(B.diag)), pB, pAB′, α′, β′)
    elseif conjA
        _diagdiagcontract!(SV(C), conj(SV(A.diag)), pA, SV(B.diag), pB, pAB′, α′, β′)
    elseif conjB
        _diagdiagcontract!(SV(C), SV(A.diag), pA, conj(SV(B.diag)), pB, pAB′, α′, β′)
    else
        _diagdiagcontract!(SV(C), SV(A.diag), pA, SV(B.diag), pB, pAB′, α′, β′)
    end
    return C
end

function tensorcontract!(
        C::Diagonal,
        A::Diagonal, pA::Index2Tuple, conjA::Bool,
        B::Diagonal, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        ::StridedNative, allocator = DefaultAllocator()
    )
    @nospecialize allocator

    # standardize input types for compilation time
    α′ = standardize_scalartype(C, α)
    β′ = standardize_scalartype(C, β)
    pAB′ = linearize(pAB)

    argcheck_tensorcontract(C, A, pA, B, pB, pAB′)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB′)

    C2 = SV(C.diag)
    if conjA && conjB
        _diagdiagdiagcontract!(C2, conj(SV(A.diag)), conj(SV(B.diag)), α′, β′)
    elseif conjA
        _diagdiagdiagcontract!(C2, conj(SV(A.diag)), SV(B.diag), α′, β′)
    elseif conjB
        _diagdiagdiagcontract!(C2, SV(A.diag), conj(SV(B.diag)), α′, β′)
    else
        _diagdiagdiagcontract!(C2, SV(A.diag), SV(B.diag), α′, β′)
    end
    return C
end

function _diagtensorcontract!(
        C::StridedView,
        A::StridedView, pA::Index2Tuple,
        Bdiag::StridedView, pB::Index2Tuple,
        pAB::IndexTuple, α::Number, β::Number
    )
    sizeA = i -> size(A, i)
    csizeA = sizeA.(pA[2])
    osizeA = sizeA.(pA[1])

    if numin(pB) == 1 # => numin(A) == numout(B) == 1
        totsize = (osizeA..., csizeA...)
        A2 = permutedims(A, linearize(pA))
        B2 = sreshape(Bdiag, ((one.(osizeA))..., csizeA...))
        C2 = permutedims(C, invperm(pAB))

    elseif numin(pB) == 0
        strideA = i -> stride(A, i)
        newstrides = (strideA.(pA[1])..., strideA(pA[2][1]) + strideA(pA[2][2]))
        totsize = (osizeA..., csizeA[1])
        A2 = StridedView(A.parent, totsize, newstrides, A.offset, A.op)
        B2 = sreshape(Bdiag, ((one.(osizeA))..., csizeA[1]))
        C2 = permutedims(C, invperm(pAB))

    else # numout(pB) == 2 # direct product
        scale!(C, β)
        β = one(β)
        A2 = sreshape(permutedims(A, linearize(pA)), (osizeA..., 1))
        B2 = sreshape(Bdiag, ((one.(osizeA))..., length(Bdiag)))

        C3 = permutedims(C, invperm(pAB))
        sC = strides(C3)
        newstrides = (Base.front(Base.front(sC))..., sC[end - 1] + sC[end])
        totsize = (osizeA..., length(Bdiag))
        C2 = StridedView(C3.parent, totsize, newstrides, C3.offset, C3.op)
    end

    Strided._mapreducedim!(Scaler(α), Adder(), Scaler(β), totsize, (C2, A2, B2))

    return C
end

function _diagdiagdiagcontract!(
        C::StridedView, Adiag::StridedView, Bdiag::StridedView, α::Number, β::Number
    )
    totsize = (length(C),)
    # required: `β` was standardized, so `Zero()` no longer kills NaNs in uninitialized `C`
    if iszero(β)
        Strided._mapreducedim!(Scaler(α), nothing, nothing, totsize, (C, Adiag, Bdiag))
    else
        Strided._mapreducedim!(Scaler(α), Adder(), Scaler(β), totsize, (C, Adiag, Bdiag))
    end
    return C
end

function _diagdiagcontract!(
        C::StridedView,
        Adiag::StridedView, pA::Index2Tuple,
        Bdiag::StridedView, pB::Index2Tuple,
        pAB::IndexTuple, α::Number, β::Number
    )
    if numin(pA) == 1 # matrix multiplication
        scale!(C, β)
        β = one(β)

        A2 = sreshape(Adiag, (length(Adiag), 1))
        B2 = sreshape(Bdiag, (length(Bdiag), 1))
        # take a view of the diagonal elements of C, having strides 1 + length(diag)
        totsize = (length(Adiag),)
        C2 = StridedView(C.parent, totsize, (sum(strides(C)),))

    elseif numin(pA) == 2 # trace
        A2 = Adiag
        B2 = Bdiag
        totsize = (length(Adiag),)
        C2 = sreshape(C, (1,))

    else # outer product
        scale!(C, β)
        β = one(β)

        A2 = sreshape(Adiag, (length(Adiag), 1))
        B2 = sreshape(Bdiag, (1, length(Adiag)))

        C3 = permutedims(C, invperm(pAB))
        strC = strides(C3)
        newstrides = (strC[1] + strC[2], strC[3] + strC[4])
        totsize = (length(A2), length(B2))
        C2 = StridedView(C3.parent, totsize, newstrides, C3.offset, C3.op)
    end

    Strided._mapreducedim!(Scaler(α), Adder(), Scaler(β), totsize, (C2, A2, B2))

    return C
end
