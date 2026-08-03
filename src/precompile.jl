using PrecompileTools: PrecompileTools, @setup_workload, @compile_workload
using Preferences: @load_preference

# Validate preferences input
# --------------------------
function validate_precompile_eltypes(eltypes)
    eltypes isa Vector{String} ||
        throw(ArgumentError("`precompile_eltypes` should be a vector of strings, got $(typeof(eltypes)) instead"))
    return map(eltypes) do Tstr
        T = eval(Meta.parse(Tstr))
        (T isa DataType && T <: Number) ||
            error("Invalid precompile_eltypes entry: `$Tstr`")
        return T
    end
end

function validate_add_ndims(add_ndims)
    add_ndims isa Int ||
        throw(ArgumentError("`precompile_add_ndims` should be an `Int`, got `$add_ndims`"))
    add_ndims ≥ 0 || error("Invalid precompile_add_ndims: `$add_ndims`")
    return add_ndims
end

function validate_trace_ndims(trace_ndims)
    trace_ndims isa Vector{Int} && length(trace_ndims) == 2 ||
        throw(ArgumentError("`precompile_trace_ndims` should be a `Vector{Int}` of length 2, got `$trace_ndims`"))
    all(≥(0), trace_ndims) || error("Invalid precompile_trace_ndims: `$trace_ndims`")
    return trace_ndims
end

function validate_contract_ndims(contract_ndims)
    contract_ndims isa Vector{Int} && length(contract_ndims) == 2 ||
        throw(ArgumentError("`precompile_contract_ndims` should be a `Vector{Int}` of length 2, got `$contract_ndims`"))
    all(≥(0), contract_ndims) ||
        error("Invalid precompile_contract_ndims: `$contract_ndims`")
    return contract_ndims
end

# Static preferences
# ------------------
const PRECOMPILE_ELTYPES = validate_precompile_eltypes(
    @load_preference("precompile_eltypes", ["Float64", "ComplexF64"])
)
const PRECOMPILE_ADD_NDIMS = validate_add_ndims(@load_preference("precompile_add_ndims", 5))
const PRECOMPILE_TRACE_NDIMS = validate_trace_ndims(
    @load_preference("precompile_trace_ndims", [4, 2])
)
const PRECOMPILE_CONTRACT_NDIMS = validate_contract_ndims(
    @load_preference("precompile_contract_ndims", [4, 2])
)

# Precompilation workload
# ------------------------
# The workload actually runs representative tensor operations so that PrecompileTools caches
# the specializations that get compiled. Each operation family is factored into a reusable
# `precompile_*` function that can also be called from a downstream package's own
# `@compile_workload` to precompile for a different backend, allocator, or array type.
#
# `@compile_workload` is enabled by default and honors the standard `precompile_workload`
# preference, which can be flipped to disable precompilation:
#
#     using TensorOperations, Preferences
#     set_preferences!(TensorOperations, "precompile_workload" => false; force=true)

# Tensor constructor used by the precompile workloads: build a rank-`N` tensor of scalar type
# `T`. Downstream callers can add methods for other array types to reuse the `precompile_*`
# functions below.
precompile_maketensor(T, N) = zeros(T, ntuple(Returns(2), N))

"""
    precompile_tensoradd(T, N, backend, allocator)

Run [`tensoradd!`](@ref) and [`tensoralloc_add`](@ref) for scalar type `T` and output rank `N`,
using `backend` and `allocator`, so that their specializations are precompiled.
"""
function precompile_tensoradd(
        T, N, backend = DefaultBackend(), allocator = DefaultAllocator()
    )
    C = precompile_maketensor(T, N)
    A = precompile_maketensor(T, N)
    pA = (ntuple(identity, N), ())

    tensoradd!(C, A, pA, false, One(), Zero(), backend, allocator)
    tensoradd!(C, A, pA, false, one(T), Zero(), backend, allocator)
    tensoradd!(C, A, pA, false, one(T), zero(T), backend, allocator)

    tensoralloc_add(T, A, pA, false, Val(true), allocator)
    tensoralloc_add(T, A, pA, false, Val(false), allocator)
    return nothing
end

"""
    precompile_tensortrace(T, (N1, N2), backend, allocator)

Run [`tensortrace!`](@ref) for scalar type `T`, output rank `N1`, and `N2` traced index pairs,
using `backend` and `allocator`, so that their specializations are precompiled.
"""
function precompile_tensortrace(
        T, (N1, N2), backend = DefaultBackend(), allocator = DefaultAllocator()
    )
    C = precompile_maketensor(T, N1)
    A = precompile_maketensor(T, N1 + 2N2)
    p = (ntuple(identity, N1), ())
    q = (ntuple(i -> N1 + i, N2), ntuple(i -> N1 + N2 + i, N2))

    tensortrace!(C, A, p, q, false, One(), Zero(), backend, allocator)
    tensortrace!(C, A, p, q, false, one(T), Zero(), backend, allocator)
    tensortrace!(C, A, p, q, false, one(T), zero(T), backend, allocator)

    # allocation re-uses tensoralloc_add
    return nothing
end

"""
    precompile_tensorcontract(T, (N1, N2, N3), backend, allocator)

Run [`tensorcontract!`](@ref) and [`tensoralloc_contract`](@ref) for scalar type `T`, with `N1`
and `N3` free output indices on the two inputs and `N2` contracted indices, using `backend` and
`allocator`, so that their specializations are precompiled.
"""
function precompile_tensorcontract(
        T, (N1, N2, N3), backend = DefaultBackend(), allocator = DefaultAllocator()
    )
    NA = N1 + N2
    NB = N2 + N3
    NC = N1 + N3
    C = precompile_maketensor(T, NC)
    A = precompile_maketensor(T, NA)
    B = precompile_maketensor(T, NB)
    pA = (ntuple(identity, N1), ntuple(i -> N1 + i, N2))
    pB = (ntuple(identity, N2), ntuple(i -> N2 + i, N3))
    pAB = (ntuple(identity, NC), ())

    tensorcontract!(C, A, pA, false, B, pB, false, pAB, One(), Zero(), backend, allocator)
    tensorcontract!(C, A, pA, false, B, pB, false, pAB, one(T), Zero(), backend, allocator)
    tensorcontract!(C, A, pA, false, B, pB, false, pAB, one(T), zero(T), backend, allocator)

    tensoralloc_contract(T, A, pA, false, B, pB, false, pAB, Val(true), allocator)
    tensoralloc_contract(T, A, pA, false, B, pB, false, pAB, Val(false), allocator)
    return nothing
end

@setup_workload begin
    @compile_workload begin
        for T in PRECOMPILE_ELTYPES
            for N in 0:PRECOMPILE_ADD_NDIMS
                precompile_tensoradd(T, N)
            end
            for N1 in 0:PRECOMPILE_TRACE_NDIMS[1], N2 in 0:PRECOMPILE_TRACE_NDIMS[2]
                precompile_tensortrace(T, (N1, N2))
            end
            for N1 in 0:PRECOMPILE_CONTRACT_NDIMS[1], N2 in 0:PRECOMPILE_CONTRACT_NDIMS[2],
                    N3 in 0:PRECOMPILE_CONTRACT_NDIMS[1]
                precompile_tensorcontract(T, (N1, N2, N3))
            end
        end
    end
end
