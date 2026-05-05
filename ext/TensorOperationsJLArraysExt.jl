module TensorOperationsJLArraysExt

using JLArrays
using TensorOperations

TensorOperations.tensoradd_type(TC, A::JLArray, pA::Index2Tuple, conjA::Bool) =
    JLArray{TC, sum(length.(pA))}

end
