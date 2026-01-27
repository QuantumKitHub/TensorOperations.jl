"""
    _flatten(ex)

Flatten nested structure of an expression, returning an unnested `Expr(:block, …)`.
"""
function _flatten(ex)
    if isa(ex, Expr) # prewalk
        ex = Expr(ex.head, map(_flatten, ex.args)...)
    end
    if isexpr(ex, :block)
        newargs = Any[]
        for e in ex.args
            if e isa Expr && e.head == :block
                append!(newargs, e.args)
            else
                push!(newargs, e)
            end
        end
        return Expr(:block, newargs...)
    elseif isexpr(ex, :(=)) && isexpr(ex.args[2], :block)
        newargs = ex.args[2].args
        newargs[end] = Expr(:(=), ex.args[1], newargs[end])
        return Expr(:block, newargs...)
    elseif isexpr(ex, :call) && ex.args[1] == :tensorscalar && isexpr(ex.args[2], :block)
        newargs = ex.args[2].args
        newargs[end] = Expr(:call, ex.args[1], newargs[end])
        return Expr(:block, newargs...)
    else
        return ex
    end
end

"""
    removelinenumbernode(ex)

Remove all `LineNumberNode`s from an expression.
"""
function removelinenumbernode(ex)
    if isexpr(ex, :block)
        args = [removelinenumbernode(e) for e in ex.args if !(e isa LineNumberNode)]
        return Expr(:block, args...)
    else
        return ex
    end
end

# list of functions that are used in expressions produced by `@tensor`
const tensoroperationsfunctions = (
    :tensoralloc, :tensorfree!,
    :tensoradd!, :tensortrace!, :tensorcontract!,
    :tensorscalar, :tensorcost, :IndexError, :scalartype,
    :checkcontractible, :promote_contract, :promote_add,
    :tensoralloc_add, :tensoralloc_contract,
    :treecost, :optimaltree, :tree2indexorder,
)
"""
    addtensoroperations(ex)

Fix references to TensorOperations functions in namespaces where `@tensor` is present but the functions are not.
"""
function addtensoroperations(ex)
    if isexpr(ex, :call) && ex.args[1] in tensoroperationsfunctions
        return Expr(
            ex.head, GlobalRef(TensorOperations, ex.args[1]),
            (addtensoroperations(ex.args[i]) for i in 2:length(ex.args))...
        )
    elseif isa(ex, Expr)
        return Expr(ex.head, (addtensoroperations(e) for e in ex.args)...)
    else
        return ex
    end
end

"""
    insertargument(ex, args, methods)

Insert an extra argument into a tensor operation, e.g. for any `op` ∈ `methods`, transform
`TensorOperations.op(args...)` -> `TensorOperations.op(args..., arg)`
"""
function insertargument(ex, arg, methods)
    if isexpr(ex, :call) && ex.args[1] isa GlobalRef &&
            ex.args[1].mod == TensorOperations && ex.args[1].name ∈ methods
        return Expr(:call, ex.args..., arg)
    elseif isa(ex, Expr)
        return Expr(ex.head, (insertargument(e, arg, methods) for e in ex.args)...)
    else
        return ex
    end
end

"""
    insertbackend(ex, backend)

Insert the backend argument into the tensor operation methods `tensoradd!`, `tensortrace!`, and `tensorcontract!`.
"""
function insertbackend(ex, backend)
    return insertargument(ex, backend, (:tensoradd!, :tensortrace!, :tensorcontract!))
end

"""
    insertallocator(ex, allocator)

Insert the allocator argument into the tensor operation and allocation methods `tensoradd!`, 
`tensortrace!`, `tensorcontract!`, `tensoralloc`, `tensoralloc_add`, `tensoralloc_contract`
and `tensorfree!`.
"""
function insertallocator(ex, allocator)
    return insertargument(
        ex, allocator,
        (
            :tensoradd!, :tensortrace!, :tensorcontract!, :tensoralloc,
            :tensoralloc_add, :tensoralloc_contract, :tensorfree!,
        )
    )
end

# TODO: this is currently only marking a single checkpoint per `@tensor` call.
"""
    insertcheckpoints(ex, allocator)

Insert the [`allocator_checkpoint!`](@ref) and [`allocator_reset!`](@ref) calls before and after tensor contractions.
"""
function insertcheckpoints(ex, allocator)
    cp = gensym("checkpoint")
    res = gensym("result")
    return quote
        $cp = $(GlobalRef(TensorOperations, :allocator_checkpoint!))($allocator)
        $res = $ex
        $(GlobalRef(TensorOperations, :allocator_reset!))($allocator, $cp)
        $res
    end
end
