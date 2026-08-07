module TensorOperationsTBLISExt

using TensorOperations
using TensorOperations: TensorOperations as TO
using TensorOperations: TBLISBackend, DefaultAllocator, Index2Tuple
using TensorOperations: StridedView, isstrided
using TensorOperations: argcheck_tensoradd, dimcheck_tensoradd,
    argcheck_tensortrace, dimcheck_tensortrace,
    argcheck_tensorcontract, dimcheck_tensorcontract
using TensorOperations: add_labels, trace_labels, contract_labels
using TensorOperations: tensoralloc_add, tensorfree!

using TBLIS
using TBLIS: len_type, stride_type, tblis_tensor

const SV = StridedView

# TBLIS only knows about these four element types, and requires all tensors taking part in a
# single operation to share it.
const TBLISFloat = Union{Float32, Float64, ComplexF32, ComplexF64}

#-------------------------------------------------------------------------------------------
# Wrapping Julia arrays as TBLIS tensors
#-------------------------------------------------------------------------------------------
# TBLIS.jl's own `tblis_tensor` constructor is restricted to `StridedArray`, which excludes
# the `StridedView`s that the tensor operations work with, and it has no way to express the
# conjugation flag. We therefore initialize the descriptor through the low-level bindings and
# patch in the flag afterwards.
for (T, init) in (
        (:Float32, :tblis_init_tensor_scaled_s),
        (:Float64, :tblis_init_tensor_scaled_d),
        (:ComplexF32, :tblis_init_tensor_scaled_c),
        (:ComplexF64, :tblis_init_tensor_scaled_z),
    )
    @eval function init_tensor!(
            p::Ptr{tblis_tensor}, A::StridedView{$T, N}, α::$T,
            len::Vector{len_type}, stride::Vector{stride_type}
        ) where {N}
        return TBLIS.$init(p, α, Cuint(N), pointer(len), pointer(A), pointer(stride))
    end
end

"""
    TBLISTensor(A::StridedView, α, isconj)

Owning counterpart of `TBLIS.tblis_tensor`, which itself only stores raw pointers into the
array it views and into the buffers holding its lengths and strides.
Rooting a `TBLISTensor` with `GC.@preserve` also roots all of those, as they are reachable
from it; pass its `ref` field wherever the library expects a `Ptr{tblis_tensor}`.

Conjugation is carried in the descriptor rather than in the type of the viewed array, so
`isconj` is an ordinary runtime flag and callers never have to branch on it to stay type
stable.
"""
struct TBLISTensor{T, N, A <: StridedView{T, N}}
    array::A
    len::Vector{len_type}
    stride::Vector{stride_type}
    ref::Base.RefValue{tblis_tensor}
end

function TBLISTensor(
        A::StridedView{T, N}, α::T, isconj::Bool
    ) where {T <: TBLISFloat, N}
    len = collect(len_type, size(A))
    stride = collect(stride_type, strides(A))
    ref = Ref{tblis_tensor}()
    GC.@preserve A len stride ref begin
        p = Base.unsafe_convert(Ptr{tblis_tensor}, ref)
        init_tensor!(p, A, α, len, stride)
        isconj && setproperty!(p, :conj, Cint(1))
    end
    return TBLISTensor{T, N, typeof(A)}(A, len, stride, ref)
end

# A `StridedView` may already carry a conjugation in its `op` field, for instance when it
# views an `Adjoint`. What TBLIS needs is the total conjugation applied to the raw data, so
# that flag and the one requested by the caller combine. For real element types `conj` is the
# identity and flagging it would only confuse the library.
hasconj(A::StridedView{T}) where {T} = T <: Complex && (A.op === conj || A.op === adjoint)
resolve_conj(A::StridedView{T}, conjA::Bool) where {T} =
    T <: Complex && (conjA ⊻ hasconj(A))

# TBLIS takes the index labels as one `char` per dimension; `add_labels` and friends hand us
# `Char` tuples that are guaranteed to be ASCII.
labels(ein::Tuple{Vararg{Char}}) = String(UInt8[c for c in ein])

#-------------------------------------------------------------------------------------------
# Argument checking
#-------------------------------------------------------------------------------------------
# Rather than quietly handing unsupported arguments to another backend, which would make a
# `backend = TBLISBackend()` that never reaches TBLIS look like it did, reject them.
@noinline function throw_eltype(f, tensors)
    types = join(eltype.(tensors), ", ")
    return throw(
        ArgumentError(
            "TBLISBackend requires all tensors of $f to share a single element type out of \
            Float32, Float64, ComplexF32 and ComplexF64, got $types"
        )
    )
end

@noinline function throw_strided(f, tensors)
    types = join(typeof.(tensors), ", ")
    return throw(
        ArgumentError("TBLISBackend requires strided arrays for $f, got $types")
    )
end

@noinline throw_conj_output(f) = throw(
    ArgumentError("TBLISBackend cannot write into a conjugated view in $f")
)

function check_arguments(f, C::AbstractArray, As::AbstractArray...)
    tensors = (C, As...)
    T = eltype(C)
    (T <: TBLISFloat && all(A -> eltype(A) === T, As)) || throw_eltype(f, tensors)
    all(isstrided, tensors) || throw_strided(f, tensors)
    hasconj(SV(C)) && throw_conj_output(f)
    return nothing
end

#-------------------------------------------------------------------------------------------
# Entry points
#-------------------------------------------------------------------------------------------
function TO.tensoradd!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::TBLISBackend, allocator = DefaultAllocator()
    )
    check_arguments(TO.tensoradd!, C, A)
    tblis_add!(SV(C), SV(A), conjA, pA, α, β)
    return C
end

function TO.tensortrace!(
        C::AbstractArray,
        A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::TBLISBackend, allocator = DefaultAllocator()
    )
    check_arguments(TO.tensortrace!, C, A)
    tblis_trace!(SV(C), SV(A), conjA, p, q, α, β)
    return C
end

function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::TBLISBackend, allocator = DefaultAllocator()
    )
    check_arguments(TO.tensorcontract!, C, A, B)
    tblis_contract!(SV(C), SV(A), pA, conjA, SV(B), pB, conjB, pAB, α, β, allocator)
    return C
end

#-------------------------------------------------------------------------------------------
# StridedView implementation
#-------------------------------------------------------------------------------------------
# `tblis_tensor_add` computes `C[einC] = β * C[einC] + α * op(A)[einA]` and does honour the
# per-tensor conjugation flag. Labels repeated within `einA` but absent from `einC` are
# traced over, so both `tensoradd!` and `tensortrace!` map onto it directly.
function tblis_add!(
        C::StridedView{T}, A::StridedView{T}, conjA::Bool, pA::Index2Tuple,
        α::Number, β::Number
    ) where {T <: TBLISFloat}
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)
    Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    einA, einC = add_labels(pA)
    return unsafe_add!(
        C, A, resolve_conj(A, conjA), einA, einC, convert(T, α), convert(T, β)
    )
end

function tblis_trace!(
        C::StridedView{T}, A::StridedView{T}, conjA::Bool,
        p::Index2Tuple, q::Index2Tuple, α::Number, β::Number
    ) where {T <: TBLISFloat}
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)
    Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    einA, einC = trace_labels(p, q)
    return unsafe_add!(
        C, A, resolve_conj(A, conjA), einA, einC, convert(T, α), convert(T, β)
    )
end

# `isconjA` is the total conjugation to apply to the raw data of `A`, as resolved by
# `resolve_conj`.
function unsafe_add!(
        C::StridedView{T}, A::StridedView{T}, isconjA::Bool, einA, einC, α::T, β::T
    ) where {T <: TBLISFloat}
    tA = TBLISTensor(A, α, isconjA)
    tC = TBLISTensor(C, β, false)
    GC.@preserve tA tC begin
        TBLIS.tblis_tensor_add(
            C_NULL, C_NULL, tA.ref, labels(einA), tC.ref, labels(einC)
        )
    end
    return C
end

function tblis_contract!(
        C::StridedView{T},
        A::StridedView{T}, pA::Index2Tuple, conjA::Bool,
        B::StridedView{T}, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number, allocator
    ) where {T <: TBLISFloat}
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)
    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    einA, einB, einC = contract_labels(pA, pB, pAB)
    α′ = convert(T, α)
    β′ = convert(T, β)
    isconjA = resolve_conj(A, conjA)
    isconjB = resolve_conj(B, conjB)

    # `tblis_tensor_mult` silently ignores the conjugation flags of its arguments (verified
    # against tblis 1.3), so conjugation has to be resolved before calling into it.
    if isconjA && isconjB
        # conj(A) * conj(B) == conj(A * B), so conjugating the output in place lets both
        # factors through unconjugated, at the cost of two passes over C. That is much
        # cheaper than materializing a conjugated copy of both A and B.
        iszero(β′) || conj!(C)
        unsafe_mult!(C, A, B, einA, einB, einC, conj(α′), conj(β′))
        conj!(C)
    elseif isconjA
        A′ = materialize_conj(A, α′, allocator)
        try
            unsafe_mult!(C, SV(A′), B, einA, einB, einC, one(T), β′)
        finally
            tensorfree!(A′, allocator)
        end
    elseif isconjB
        B′ = materialize_conj(B, one(T), allocator)
        try
            unsafe_mult!(C, A, SV(B′), einA, einB, einC, α′, β′)
        finally
            tensorfree!(B′, allocator)
        end
    else
        unsafe_mult!(C, A, B, einA, einB, einC, α′, β′)
    end
    return C
end

function unsafe_mult!(
        C::StridedView{T}, A::StridedView{T}, B::StridedView{T},
        einA, einB, einC, α::T, β::T
    ) where {T <: TBLISFloat}
    # TBLIS scales the product by the scalars of both factors, so α rides along on A alone.
    # The conjugation flags are all `false` because `tblis_tensor_mult` ignores them and the
    # caller has resolved the conjugations already.
    tA = TBLISTensor(A, α, false)
    tB = TBLISTensor(B, one(T), false)
    tC = TBLISTensor(C, β, false)
    GC.@preserve tA tB tC begin
        TBLIS.tblis_tensor_mult(
            C_NULL, C_NULL, tA.ref, labels(einA), tB.ref, labels(einB),
            tC.ref, labels(einC)
        )
    end
    return C
end

# Write `α * conj(raw A)` into a fresh temporary that keeps the index order of `A`, so that
# the labels computed for `A` remain valid for it and it can be fed to `mult` unconjugated.
function materialize_conj(A::StridedView{T, N}, α::T, allocator) where {T <: TBLISFloat, N}
    pA = (ntuple(identity, N), ())
    A′ = tensoralloc_add(T, A, pA, false, Val(true), allocator)
    einA, einC = add_labels(pA)
    unsafe_add!(SV(A′), A, true, einA, einC, α, zero(T))
    return A′
end

end # module TensorOperationsTBLISExt
