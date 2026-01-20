module TensorOperations

using VectorInterface
using TupleTools: TupleTools, isperm, invperm

using LinearAlgebra
using LinearAlgebra: mul!, BlasFloat
using Strided
using StridedViews: isstrided
using PtrArrays
using LRUCache

using Base.Meta: isexpr

# Exports
#---------
# export macro API
export @tensor, @tensoropt, @tensoropt_verbose, @optimalcontractiontree, @notensor, @ncon
export @cutensor, @butensor

# export function based API
export ncon
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!, tensorscalar
export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalartype
export tensoralloc, tensorfree!

export IndexTuple, Index2Tuple, linearize

# export debug functionality
export checkcontractible, tensorcost

# Interface and index types
#---------------------------
include("indices.jl")
include("backends.jl")
include("interface.jl")
include("utils.jl")

# Generic pullbacks for AD
#---------------------------
include("pullbacks/add.jl")
include("pullbacks/trace.jl")
include("pullbacks/contract.jl")

# Index notation via macros
#---------------------------
@nospecialize
include("indexnotation/verifiers.jl")
include("indexnotation/analyzers.jl")
include("indexnotation/preprocessors.jl")
include("indexnotation/postprocessors.jl")
include("indexnotation/contractiontrees.jl")
include("indexnotation/ncontree.jl")
include("indexnotation/indexordertree.jl")
include("indexnotation/poly.jl")
include("indexnotation/optdata.jl")
include("indexnotation/optimaltree.jl")
include("indexnotation/instantiators.jl")
include("indexnotation/parser.jl")
include("indexnotation/tensormacros.jl")
@specialize

# Implementations
#-----------------
include("implementation/functions.jl")
include("implementation/ncon.jl")
include("implementation/abstractarray.jl")
include("implementation/strided.jl")
include("implementation/blascontract.jl")
include("implementation/diagonal.jl")
include("implementation/base.jl")
include("implementation/indices.jl")
include("implementation/allocator.jl")

# Global variables
#------------------
const costcache = LRU{Any, Any}(; maxsize = 10^5)

# Package extensions backwards compatibility
#--------------------------------------------

using PackageExtensionCompat
function __init__()
    @require_extensions
    return nothing
end

include("precompile.jl")

end # module
