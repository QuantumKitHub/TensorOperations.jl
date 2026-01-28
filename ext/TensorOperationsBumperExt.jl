module TensorOperationsBumperExt

using TensorOperations
using Bumper

const BumperBuffer = Union{SlabBuffer, AllocBuffer}

function TensorOperations.tensoralloc(
        ::Type{A}, structure, ::Val{istemp}, buf::BumperBuffer
    ) where {A <: AbstractArray, istemp}
    # TODO: remove the `ndims` check if this is fixed in Bumper / StrideArraysCore
    if istemp && ndims(A) > 0
        return Bumper.alloc!(buf, eltype(A), structure...)
    else
        return TensorOperations.tensoralloc(A, structure, Val(istemp))
    end
end

TensorOperations.allocator_checkpoint!(alloc::BumperBuffer) = Bumper.checkpoint_save(alloc)
TensorOperations.allocator_reset!(::BumperBuffer, cp) = Bumper.checkpoint_restore!(cp)

function TensorOperations._butensor(src, ex...)
    buf_sym = gensym("buffer")

    # TODO: there is no check for doubled tensor kwargs
    newex = quote
        $buf_sym = $(Expr(:call, GlobalRef(Bumper, :default_buffer)))
        $(
            Expr(
                :macrocall, GlobalRef(TensorOperations, Symbol("@tensor")),
                src, :(allocator = $buf_sym), ex...
            )
        )
    end
    return Base.remove_linenums!(newex)
end

end
