# the Windows build of `tblis_jll` aborts the process from inside the library, so
# the extension refuses to call into it and there is nothing else to test
using TensorOperations
using TBLIS
using Test

A = randn(Float64, (3, 4))
@test_throws ArgumentError tensoradd!(
    zeros(Float64, (4, 3)), A, ((2, 1), ()), false, 1, 0,
    TensorOperations.TBLISBackend()
)
