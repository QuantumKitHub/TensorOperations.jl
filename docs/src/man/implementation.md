```@meta
CurrentModule = TensorOperations
```

# Implementation

```@meta
CurrentModule = TensorOperations
```

The [`@tensor`](@ref) macro and its relatives work as parsers for indexed tensor
expressions. They transform these into a sequence of calls to the primitive tensor
operations. This allows the support of custom types that implement the [Interface](@ref).
The actual implementation is achieved through the use of `TensorParser`, which provides the
general framework to parse tensor expressions. The `@tensor` macro is then just a wrapper
around this, which configures the default behavior and handles keyword arguments of the
parser.

The `TensorParser` works by breaking down the parsing into three main phases. First, a basic
check of the supplied expression is performed, to ensure that it is a valid tensor
expression. Then, a number of preprocessing steps can be performed, which are used to
standardize expressions, allow for syntactic sugar features, and can also be used as a hook
for writing custom parsers. Then, the different contractions within the tensor expression
are analyzed and processed, which rewrites the expression into a set of binary rooted trees.
Then, the main step can be executed, namely transforming the whole expression into actual
calls to the primitive tensor operations [`tensoradd!`](@ref), [`tensortrace!`](@ref) and
[`tensorcontract!`](@ref), as well as calls to [`tensoralloc_add`](@ref) and
[`tensoralloc_contract`](@ref) to allocate the temporary and final tensors. For those, also
the resulting scalar type needs to be determined. Finally, a number of postprocessing steps
can be added, which are mostly used to clean up the resulting expression by flattening and
by removing line number nodes, but also to incorporate the custom backend and allocation
system.

## Verifiers

The basic checks are performed by [`verifytensorexpr`](@ref), which calls the verifiers
[`isassignment`](@ref), [`isdefinition`](@ref), [`istensor`](@ref), [`istensorexpr`](@ref)
and [`isscalarexpr`](@ref).

```@docs
TensorOperations.verifytensorexpr
TensorOperations.isassignment
TensorOperations.isdefinition
TensorOperations.isindex
TensorOperations.istensor
TensorOperations.istensorexpr
TensorOperations.isscalarexpr
```

## Preprocessing

The following functions exist as preprocessors and are enabled in the default `TensorParser`
objects.

```@docs
TensorOperations.normalizeindices
TensorOperations.expandconj
TensorOperations.nconindexcompletion
TensorOperations.extracttensorobjects
TensorOperations.insertcontractionchecks
```

## Processing

The following functions implement the main steps in parsing the tensor expression, and are
always performed by any `TensorParser` object.

```@docs
TensorOperations.processcontractions
TensorOperations.tensorify
```

## Postprocessing

The following functions exist as postprocessors and are enabled in the default `TensorParser`
objects.

```@docs
TensorOperations._flatten
TensorOperations.addtensoroperations
TensorOperations.insertargument
TensorOperations.insertbackend
TensorOperations.insertallocator
```

Finally, after all postprocessors have run, the parser strips the `LineNumberNode`s that it
synthesized itself, while preserving the ones that came from the user's code. As a result the
generated code stays attributable to the lines of the original `@tensor` expression, which is
what makes those lines show up in code coverage reports and in stacktraces.

## Analysis of contraction graphs and optimizing contraction order

The macro [`@tensoropt`](@ref) or the combination of [`@tensor`](@ref) with the keyword
`opt` can be used to optimize the contraction order of the expression at compile time. This
is done by analyzing the contraction graph, where the nodes are the tensors and the edges
are the contractions, in combination with the data provided in `optdata`, which is a
dictionary associating a cost (either a number or a polynomial in some abstract scaling
parameter) to every index. This information is then used to determine the (asymptotically)
optimal contraction tree (in terms of number of floating point operations). The algorithm
that is used is described in [arXiv:1304.6112](https://arxiv.org/abs/1304.6112).
