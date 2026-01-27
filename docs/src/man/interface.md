# Interface

The `@tensor` macro rewrites tensor operations in terms of basic building blocks, such that
any tensor type that implements the following interface can be supported. In these methods,
`C` will indicate an output tensor which is changed in-place, while `A` and `B` denote input
tensors that are left unaltered. `pC`, `pA` and `pB` denote an `Index2Tuple`, a tuple of two
tuples that represents a permutation and partition of the original tensor indices. Finally,
`conjA` and `conjB` are boolean values that are used to indicate if the input tensor should be
conjugated (`true`) or used as-is (`false`).

## Operations

The three primitive tensor operations have already been described in the previous section,
and correspond to

* [`tensoradd!`](@ref),
* [`tensortrace!`](@ref),
* [`tensorcontract!`](@ref).

All other functions described in the previous section are implemented in terms of those. 
Hence, those are the only methods that need to be overloaded to support e.g. a new tensor 
type of to implement a new backend and/or allocator. For new types of tensors, it is
possible to implement the methods without the final two arguments `backend` and `allocator`,
if you know that you will never need them. They will not be inserted by the `@tensor` macro
if they are not explicitly specified as keyword arguments. However, it is possible to add
those arguments with the default values and ignore them in case they are not needed.

Alternatively, if some new tensor type is backed by an `AbstractArray` instance, and the
tensor operations are also implemented by applying the same operations to the underlying
array, it is possible to forward the value of the `backend` and `allocator` arguments in
order to still support the full flexibility of the `@tensor` macro.

* [`TensorOperations.DefaultBackend`](@ref)
* [`TensorOperations.DefaultAllocator`](@ref)

There is one more necessary tensor operation, which is to convert back from a rank zero
tensor to a scalar quantity. 

```@docs
tensorscalar
```

As there is typically a simple and unique way of doing so, this method does not have any
`backend` or `allocator` arguments.

## Allocations

For some networks, it will be necessary to allocate output and/or intermediate tensors. This
process is split into the following hierarchy of functions, where custom tensor types can
opt in at different levels.

By default, the process is split into three steps.

First, the scalar type `TC` of the resulting tensor is determined. This is done by
leveraging [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl)'s `scalartype`,
and promoting the results along with the types of any scalars that are present.

```@docs
TensorOperations.promote_add
TensorOperations.promote_contract
```

Then, the type and structure of the resulting tensor is determined. The former represents
all the information that is contained within the type, while the latter adds the required
runtime information (e.g. array sizes, ...).

```@docs
TensorOperations.tensorstructure
TensorOperations.tensoradd_type
TensorOperations.tensoradd_structure
TensorOperations.tensorcontract_type
TensorOperations.tensorcontract_structure
```

Finally, the tensor is allocated, where a flag indicates if this is a temporary object, or
one that will persist outside of the scope of the macro. If the resulting tensor is a
temporary object and its memory will not be freed by Julia's garbage collector, it can be
explicitly freed by implementing `tensorfree!`, which by default does nothing.

```@docs
tensoralloc
tensorfree!
```

These functions also depend on the optional `allocator` argument that can be used to
control different allocation strategies, and for example to differentiate between
allocations of temporary tensors as opposed to tensors that are part of the output.

The `@tensor` macro will however only insert the calls to the following functions, which
have a default implementation in terms of the functions above.

```@docs
TensorOperations.tensoralloc_add
TensorOperations.tensoralloc_contract
```

For allocators that manage reusable buffers or maintain state across multiple tensor
operations, the following helper functions provide an interface for managing allocation
regions:

```@docs
TensorOperations.allocator_checkpoint!
TensorOperations.allocator_reset!
```

These can be used to save the current allocator state before a region of code and restore it
afterwards, where it is guaranteed that all temporary objects allocated within that region
are released. This is particularly useful for nested tensor operations or when repeatedly
evaluating the same tensor contraction with an allocator like
[`TensorOperations.BufferAllocator`](@ref).

## Utility

Some of the optional keywords for `@tensor` can be accessed only after implementing the
following utility functions:

```@docs
tensorcost
checkcontractible
```
