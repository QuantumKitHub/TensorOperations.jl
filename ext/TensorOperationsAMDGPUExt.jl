module TensorOperationsAMDGPUExt

using AMDGPU
using TensorOperations
using TensorOperations: TensorOperations as TO

#-------------------------------------------------------------------------------------------
# Allocator
#-------------------------------------------------------------------------------------------

TO.tensoradd_type(TC, A::AnyRocArray, pA::Index2Tuple, conjA::Bool) =
    ROCArray{TC, TO.numind(pA)}

function TO.tensoralloc_add(
        TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        istemp::Val, allocator::TO.AMDAllocator
    )
    ttype = ROCArray{TC, TO.numind(pA)}
    structure = TO.tensoradd_structure(A, pA, conjA)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

function TO.tensoralloc_contract(
        TC,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        istemp::Val, allocator::TO.AMDAllocator
    )
    ttype = ROCArray{TC, TO.numind(pAB)}
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

# NOTE: the general implementation in the `DefaultAllocator` case works just fine, without
# selecting an explicit memory model
function TO.tensoralloc(
        ::Type{<:ROCArray{T, N}}, structure,
        ::Val{istemp}, allocator::TO.AMDAllocator
    ) where {T, N}
    return ROCArray{T, N}(undef, structure)
end

function TO.tensorfree!(C::ROCArray, ::TO.AMDAllocator)
    AMDGPU.unsafe_free!(C)
    return nothing
end

end
