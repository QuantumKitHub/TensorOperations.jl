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
# the `StridedView`s used below, and it has no way to express the conjugation flag. We
# therefore initialize the descriptor through the low-level bindings and patch in the flag
# afterwards.
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

isconj(A::StridedView{T}, conjA::Bool) where {T} = T <: Complex && (conjA ⊻ (A.op === conj))
tblis_dims(A::StridedView) = (collect(len_type, size(A)), collect(stride_type, strides(A)))

"""
    tblis_tensor(A::StridedView, α, len, stride, conj) -> Ref{TBLIS.tblis_tensor}

Descriptor for `α * A`, conjugated when `conj` is set, using `len` and `stride` as the
buffers handed to TBLIS.

The descriptor only stores raw pointers into `A`, `len` and `stride`, so all three, along
with the returned `Ref`, have to be kept alive by the caller for as long as TBLIS may access
them. Note that `conj` is the *total* conjugation applied to the data of `A`, as computed by
[`isconj`](@ref), not the flag the caller was handed.
"""
function tblis_tensor(
        A::StridedView{T, N}, α::T,
        len::Vector{len_type}, stride::Vector{stride_type}, conj::Bool
    ) where {T <: TBLISFloat, N}
    ref = Ref{tblis_tensor}()
    GC.@preserve A len stride ref begin
        p = Base.unsafe_convert(Ptr{tblis_tensor}, ref)
        init_tensor!(p, A, α, len, stride)
        conj && setproperty!(p, :conj, Cint(1))
    end
    return ref
end

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

# A conjugated output cannot be expressed: TBLIS applies the flag of `C` when reading the
# `β * C` term but not when writing the result back, so it would conjugate half the
# operation. Hence `check_arguments` rejects such a `C` and every `C` descriptor below is
# built unconjugated.
@noinline throw_conj_output(f) = throw(
    ArgumentError("TBLISBackend cannot write into a conjugated view in $f")
)

function check_arguments(f, C::AbstractArray, As::AbstractArray...)
    tensors = (C, As...)
    T = eltype(C)
    (T <: TBLISFloat && all(A -> eltype(A) === T, As)) || throw_eltype(f, tensors)
    all(isstrided, tensors) || throw_strided(f, tensors)
    isconj(SV(C), false) && throw_conj_output(f)
    return nothing
end

#-------------------------------------------------------------------------------------------
# Operations
#-------------------------------------------------------------------------------------------
# `tblis_tensor_add` computes `C[einC] = β * C[einC] + α * op(A)[einA]` and does honour the
# per-tensor conjugation flag. Labels repeated within `einA` but absent from `einC` are traced
# over, so both `tensoradd!` and `tensortrace!` map onto it directly.
function TO.tensoradd!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::TBLISBackend, allocator = DefaultAllocator()
    )
    check_arguments(TO.tensoradd!, C, A)
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)
    Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    T = eltype(C)
    einA, einC = add_labels(pA)
    # `SV` supports inputs such as `Adjoint`, which have no `strides` or `pointer` of their own
    Av, Cv = SV(A), SV(C)
    lenA, strideA = tblis_dims(Av)
    lenC, strideC = tblis_dims(Cv)
    GC.@preserve Av Cv lenA strideA lenC strideC begin
        tA = tblis_tensor(Av, convert(T, α), lenA, strideA, isconj(Av, conjA))
        tC = tblis_tensor(Cv, convert(T, β), lenC, strideC, false)
        TBLIS.tblis_tensor_add(C_NULL, C_NULL, tA, labels(einA), tC, labels(einC))
    end
    return C
end

function TO.tensortrace!(
        C::AbstractArray,
        A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::TBLISBackend, allocator = DefaultAllocator()
    )
    check_arguments(TO.tensortrace!, C, A)
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)
    Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    T = eltype(C)
    einA, einC = trace_labels(p, q)
    # `SV` supports inputs such as `Adjoint`, which have no `strides` or `pointer` of their own
    Av, Cv = SV(A), SV(C)
    lenA, strideA = tblis_dims(Av)
    lenC, strideC = tblis_dims(Cv)
    GC.@preserve Av Cv lenA strideA lenC strideC begin
        tA = tblis_tensor(Av, convert(T, α), lenA, strideA, isconj(Av, conjA))
        tC = tblis_tensor(Cv, convert(T, β), lenC, strideC, false)
        TBLIS.tblis_tensor_add(C_NULL, C_NULL, tA, labels(einA), tC, labels(einC))
    end
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
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)
    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    T = eltype(C)
    einA, einB, einC = contract_labels(pA, pB, pAB)
    α′ = convert(T, α)
    β′ = convert(T, β)
    # `SV` supports inputs such as `Adjoint`, which have no `strides` or `pointer` of their own
    Av, Bv, Cv = SV(A), SV(B), SV(C)
    isconjA = isconj(Av, conjA)
    isconjB = isconj(Bv, conjB)

    # `tblis_tensor_mult` silently ignores the conjugation flags of its arguments (verified
    # against tblis 1.3), so conjugation has to be resolved before calling into it.
    if isconjA && isconjB
        # conj(A) * conj(B) == conj(A * B), so conjugating the output in place lets both
        # factors through unconjugated, at the cost of two passes over C. That is much
        # cheaper than materializing a conjugated copy of both A and B.
        iszero(β′) || conj!(Cv)
        tblis_mult!(Cv, Av, Bv, einA, einB, einC, conj(α′), conj(β′))
        conj!(Cv)
    elseif isconjA
        A′ = materialize_conj(Av, conjA, α′, allocator)
        tblis_mult!(Cv, SV(A′), Bv, einA, einB, einC, one(T), β′)
        tensorfree!(A′, allocator)
    elseif isconjB
        B′ = materialize_conj(Bv, conjB, one(T), allocator)
        tblis_mult!(Cv, Av, SV(B′), einA, einB, einC, α′, β′)
        tensorfree!(B′, allocator)
    else
        tblis_mult!(Cv, Av, Bv, einA, einB, einC, α′, β′)
    end
    return C
end

# Shared by the four conjugation branches above, which have already folded any conjugation
# into the data, so the descriptors are built unconjugated. TBLIS scales the product by the
# scalars of both factors, so α rides along on A alone.
function tblis_mult!(
        C::StridedView{T}, A::StridedView{T}, B::StridedView{T},
        einA, einB, einC, α::T, β::T
    ) where {T <: TBLISFloat}
    lenA, strideA = tblis_dims(A)
    lenB, strideB = tblis_dims(B)
    lenC, strideC = tblis_dims(C)
    GC.@preserve A B C lenA strideA lenB strideB lenC strideC begin
        tA = tblis_tensor(A, α, lenA, strideA, false)
        tB = tblis_tensor(B, one(T), lenB, strideB, false)
        tC = tblis_tensor(C, β, lenC, strideC, false)
        TBLIS.tblis_tensor_mult(
            C_NULL, C_NULL, tA, labels(einA), tB, labels(einB), tC, labels(einC)
        )
    end
    return C
end

# Write `α * conj(A)` into a fresh temporary that keeps the index order of `A`, so that the
# labels computed for `A` remain valid for it and it can be fed to `mult` unconjugated.
function materialize_conj(
        A::StridedView{T, N}, conjA::Bool, α::T, allocator
    ) where {T <: TBLISFloat, N}
    pA = (ntuple(identity, N), ())
    A′ = tensoralloc_add(T, A, pA, false, Val(true), allocator)
    TO.tensoradd!(A′, A, pA, conjA, α, zero(T), TBLISBackend(), allocator)
    return A′
end

end # module TensorOperationsTBLISExt
