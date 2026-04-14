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
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA
        stridedtensoradd!(SV(C), conj(SV(A)), pA, α, β, backend, allocator)
    else
        stridedtensoradd!(SV(C), SV(A), pA, α, β, backend, allocator)
    end
    return C
end

function tensortrace!(
        C::AbstractArray,
        A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::StridedBackend, allocator = DefaultAllocator()
    )
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA
        stridedtensortrace!(SV(C), conj(SV(A)), p, q, α, β, backend, allocator)
    else
        stridedtensortrace!(SV(C), SV(A), p, q, α, β, backend, allocator)
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
        C::StridedView,
        A::StridedView, pA::Index2Tuple,
        α::Number, β::Number,
        ::StridedBackend, allocator = DefaultAllocator()
    )
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)
    if !istrivialpermutation(pA) && Base.mightalias(C, A)
        throw(ArgumentError("output tensor must not be aliased with input tensor"))
    end

    A′ = permutedims(A, linearize(pA))
    Strided._mapreducedim!(Scaler(α), Adder(), Scaler(β), size(C), (C, A′))
    return C
end

function stridedtensortrace!(
        C::StridedView,
        A::StridedView, p::Index2Tuple, q::Index2Tuple,
        α::Number, β::Number,
        ::StridedBackend, allocator = DefaultAllocator()
    )
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)

    Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    sizeA = i -> size(A, i)
    strideA = i -> stride(A, i)
    tracesize = sizeA.(q[1])
    newstrides = (strideA.(linearize(p))..., (strideA.(q[1]) .+ strideA.(q[2]))...)
    newsize = (size(C)..., tracesize...)

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
