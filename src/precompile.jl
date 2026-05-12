using PrecompileTools: PrecompileTools
using Preferences: @load_preference, load_preference

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

# Copy from PrecompileTools.workload_enabled but default to false
function workload_enabled(mod::Module = @__MODULE__)
    return try
        if load_preference(PrecompileTools, "precompile_workloads", true)
            return load_preference(mod, "precompile_workload", false)
        else
            return false
        end
    catch
        false
    end
end

# Using explicit precompile statements here instead of @compile_workload:
# Actually running the precompilation through PrecompileTools leads to longer compile times
# Keeping the workload_enabled functionality to have the option of disabling precompilation
# in a compatible manner with the rest of the ecosystem
if workload_enabled()
    # tensoradd!
    # ----------
    for T in PRECOMPILE_ELTYPES
        for N in 0:PRECOMPILE_ADD_NDIMS
            C = Array{T, N}
            A = Array{T, N}
            pA = Index2Tuple{N, 0}

            precompile(tensoradd!, (C, A, pA, Bool, One, Zero))
            precompile(tensoradd!, (C, A, pA, Bool, T, Zero))
            precompile(tensoradd!, (C, A, pA, Bool, T, T))

            precompile(tensoralloc_add, (T, A, pA, Bool, Val{true}))
            precompile(tensoralloc_add, (T, A, pA, Bool, Val{false}))
        end
    end

    # tensortrace!
    # ------------
    for T in PRECOMPILE_ELTYPES
        for N1 in 0:PRECOMPILE_TRACE_NDIMS[1], N2 in 0:PRECOMPILE_TRACE_NDIMS[2]
            C = Array{T, N1}
            A = Array{T, N1 + 2N2}
            p = Index2Tuple{N1, 0}
            q = Index2Tuple{N2, N2}

            precompile(tensortrace!, (C, A, p, q, Bool, One, Zero))
            precompile(tensortrace!, (C, A, p, q, Bool, T, Zero))
            precompile(tensortrace!, (C, A, p, q, Bool, T, T))

            # allocation re-uses tensoralloc_add
        end
    end

    # tensorcontract!
    # ---------------
    for T in PRECOMPILE_ELTYPES
        for N1 in 0:PRECOMPILE_CONTRACT_NDIMS[1], N2 in 0:PRECOMPILE_CONTRACT_NDIMS[2],
                N3 in 0:PRECOMPILE_CONTRACT_NDIMS[1]

            NA = N1 + N2
            NB = N2 + N3
            NC = N1 + N3
            C, A, B = Array{T, NC}, Array{T, NA}, Array{T, NB}
            pA = Index2Tuple{N1, N2}
            pB = Index2Tuple{N2, N3}
            pAB = Index2Tuple{NC, 0}

            precompile(tensorcontract!, (C, A, pA, Bool, B, pB, Bool, pAB, One, Zero))
            precompile(tensorcontract!, (C, A, pA, Bool, B, pB, Bool, pAB, T, Zero))
            precompile(tensorcontract!, (C, A, pA, Bool, B, pB, Bool, pAB, T, T))

            precompile(tensoralloc_contract, (T, A, pA, Bool, B, pB, Bool, pAB, Val{true}))
            precompile(tensoralloc_contract, (T, A, pA, Bool, B, pB, Bool, pAB, Val{false}))
        end
    end
end
