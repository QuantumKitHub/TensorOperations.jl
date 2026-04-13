module TensorOperationsCUDAExt

using CUDA
using TensorOperations
using TensorOperations: TensorOperations as TO

#-------------------------------------------------------------------------------------------
# Allocator
#-------------------------------------------------------------------------------------------

TO.tensoradd_type(TC, A::CuArray, pA::Index2Tuple, conjA::Bool) =
    CuArray{TC, TO.numind(pA)}

function TO.CUDAAllocator()
    Mout = CUDA.UnifiedMemory
    Min = CUDA.default_memory
    Mtemp = CUDA.default_memory
    return TO.CUDAAllocator{Mout, Min, Mtemp}()
end

function TO.tensoralloc_add(
        TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        istemp::Val, allocator::TO.CUDAAllocator
    )
    ttype = CuArray{TC, TO.numind(pA)}
    structure = TO.tensoradd_structure(A, pA, conjA)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

function TO.tensoralloc_contract(
        TC,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        istemp::Val, allocator::TO.CUDAAllocator
    )
    ttype = CuArray{TC, TO.numind(pAB)}
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

# NOTE: the general implementation in the `DefaultAllocator` case works just fine, without
# selecting an explicit memory model
function TO.tensoralloc(
        ::Type{CuArray{T, N}}, structure,
        ::Val{istemp}, allocator::TO.CUDAAllocator{Mout, Min, Mtemp}
    ) where {T, N, istemp, Mout, Min, Mtemp}
    M = istemp ? Mtemp : Mout
    return CuArray{T, N, M}(undef, structure)
end

function TO.tensorfree!(C::CuArray, ::TO.CUDAAllocator)
    CUDA.unsafe_free!(C)
    return nothing
end

end
