# Automatic differentiation

TensorOperations offers experimental support for reverse-mode automatic diffentiation (AD)
through the use of [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)
and [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl). As the basic
operations are multi-linear, the vector-Jacobian products thereof can all be expressed in
terms of the operations defined in VectorInterface and TensorOperations. Thus, any custom
type whose tangent type also support these interfaces will automatically inherit
reverse-mode AD support.

As the [`@tensor`](@ref) macro rewrites everything in terms of the basic tensor operations,
the reverse-mode rules for these methods are supplied. However, because ChainRules.jl does
not support in-place mutation, effectively these operations will be replaced with a
non-mutating version. This is similar to the behaviour found in
[BangBang.jl](https://github.com/JuliaFolds/BangBang.jl), as the operations will be
in-place, except for the pieces of code that are being differentiated. In effect, this
amounts to replacing all assignments (`=`) with definitions (`:=`) within the context of
[`@tensor`](@ref).

Mooncake.jl *does* support in-place mutation, and as a result on the reverse pass
all mutated input variables should be restored to their state before the forward-pass
function was called. Currently, this is **not done** for buffers you provide to various
TensorOperations functions, so relying on the state of the buffer (e.g. a bumper) being
restored will **silently** return incorrect results.

!!! warning "Experimental"

    While some rudimentary tests are run, the AD support is currently not incredibly
    well-tested. Because of the way it is implemented, the use of AD will tacitly replace
    mutating operations with a non-mutating variant. This might lead to unwanted bugs that
    are hard to track down. Additionally, for mixed scalar types their also might be
    unexpected or unwanted behaviour.
