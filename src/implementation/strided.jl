const StridedViewOrDiagonal = Union{StridedView, Diagonal}

select_backend(::typeof(tensoradd!), C::StridedView, A::StridedView) = StridedNative()
select_backend(::typeof(tensortrace!), C::StridedView, A::StridedView) = StridedNative()

select_backend(::typeof(tensorcontract!), C::StridedView, A::StridedView, B::StridedView) =
    eltype(C) <: LinearAlgebra.BlasFloat ? StridedBLAS() : StridedNative()
select_backend(::typeof(tensorcontract!), C::StridedViewOrDiagonal, A::StridedViewOrDiagonal, B::StridedViewOrDiagonal) =
    StridedNative()

#-------------------------------------------------------------------------------------------
# Force strided implementation on AbstractArray instances with Strided backend
#-------------------------------------------------------------------------------------------
const SV = StridedView
function tensoradd!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::StridedBackend, allocator = DefaultAllocator()
    )
    @nospecialize backend allocator

    # standardize input types for compilation time
    α′ = standardize_scalartype(C, α)
    β′ = standardize_scalartype(C, β)
    p = linearize(pA)

    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA
        stridedtensoradd!(SV(C), conj(SV(A)), p, α′, β′)
    else
        stridedtensoradd!(SV(C), SV(A), p, α′, β′)
    end

    return C
end

function tensortrace!(
        C::AbstractArray,
        A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::StridedBackend, allocator = DefaultAllocator()
    )
    @nospecialize backend allocator

    # standardize input types for compilation time
    α′ = standardize_scalartype(C, α)
    β′ = standardize_scalartype(C, β)
    p′ = linearize(p)

    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA
        stridedtensortrace!(SV(C), conj(SV(A)), p′, q, α′, β′)
    else
        stridedtensortrace!(SV(C), SV(A), p′, q, α′, β′)
    end

    return C
end

function tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::StridedBackend, allocator = DefaultAllocator()
    )
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA && conjB
        stridedtensorcontract!(
            SV(C), conj(SV(A)), pA, conj(SV(B)), pB, pAB, α, β, backend, allocator
        )
    elseif conjA
        stridedtensorcontract!(
            SV(C), conj(SV(A)), pA, SV(B), pB, pAB, α, β, backend, allocator
        )
    elseif conjB
        stridedtensorcontract!(
            SV(C), SV(A), pA, conj(SV(B)), pB, pAB, α, β, backend, allocator
        )
    else
        stridedtensorcontract!(
            SV(C), SV(A), pA, SV(B), pB, pAB, α, β, backend, allocator
        )
    end
    return C
end

#-------------------------------------------------------------------------------------------
# StridedView implementation
#-------------------------------------------------------------------------------------------
struct Adder end
(::Adder)(x, y) = VectorInterface.add(x, y)
struct Scaler{T}
    α::T
end
(s::Scaler)(x) = scale(x, s.α)
(s::Scaler)(x, y) = scale(x * y, s.α)

function stridedtensoradd!(
        C::StridedView, A::StridedView, pA::IndexTuple, α::Number, β::Number,
    )
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)
    !istrivialpermutation(pA) && Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))
    Ap = permutedims(A, pA)
    if iszero(β)
        Strided._mapreducedim!(Scaler(α), nothing, nothing, size(C), (C, Ap))
    else
        Strided._mapreducedim!(Scaler(α), Adder(), Scaler(β), size(C), (C, Ap))
    end
    return C
end

function stridedtensortrace!(
        C::StridedView, A::StridedView, p::IndexTuple, q::Index2Tuple, α::Number, β::Number,
    )
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)
    Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))
    newsize = linearize(size(C), TupleTools.getindices(size(A), q[1]))
    stA = strides(A)
    newstrides = linearize(
        TupleTools.getindices(stA, p),
        TupleTools.getindices(stA, q[1]) .+ TupleTools.getindices(stA, q[2])
    )
    A′ = SV(A.parent, newsize, newstrides, A.offset, A.op)
    Strided._mapreducedim!(Scaler(α), Adder(), Scaler(β), newsize, (C, A′))
    return C
end

function stridedtensorcontract!(
        C::StridedView,
        A::StridedView, pA::Index2Tuple,
        B::StridedView, pB::Index2Tuple,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::StridedBLAS, allocator = DefaultAllocator()
    )
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    blas_contract!(C, A, pA, B, pB, pAB, α, β, backend, allocator)
    return C
end

function stridedtensorcontract!(
        C::StridedView,
        A::StridedView, pA::Index2Tuple,
        B::StridedView, pB::Index2Tuple,
        pAB::Index2Tuple,
        α::Number, β::Number,
        ::StridedNative, allocator = DefaultAllocator()
    )
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    sizeA = size(A)
    sizeB = size(B)
    csizeA = TupleTools.getindices(sizeA, pA[2])
    csizeB = TupleTools.getindices(sizeB, pB[1])
    osizeA = TupleTools.getindices(sizeA, pA[1])
    osizeB = TupleTools.getindices(sizeB, pB[2])

    AS = sreshape(permutedims(A, linearize(pA)), (osizeA..., one.(osizeB)..., csizeA...))
    BS = sreshape(
        permutedims(B, linearize(reverse(pB))),
        (one.(osizeA)..., osizeB..., csizeB...)
    )
    CS = sreshape(
        permutedims(C, invperm(linearize(pAB))),
        (osizeA..., osizeB..., one.(csizeA)...)
    )
    tsize = (osizeA..., osizeB..., csizeA...)

    Strided._mapreducedim!(Scaler(α), Adder(), Scaler(β), tsize, (CS, AS, BS))
    return C
end
