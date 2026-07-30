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
    linenumberfiles(ex, files = Set{Symbol}())

Collect the set of files referenced by the `LineNumberNode`s in `ex`.

This is used on the expression that is handed to a [`TensorParser`](@ref), before any
processing takes place, to determine which `LineNumberNode`s belong to the user's code:
see [`removeinternallinenumbernodes`](@ref).
"""
function linenumberfiles(ex, files = Set{Symbol}())
    if ex isa LineNumberNode
        push!(files, ex.file)
    elseif ex isa Expr
        foreach(e -> linenumberfiles(e, files), ex.args)
    end
    return files
end

"""
    removeinternallinenumbernodes(ex, userfiles)

Remove the `LineNumberNode`s that were synthesized by the parser, i.e. the ones whose file is
not in `userfiles`, as obtained from [`linenumberfiles`](@ref) on the original expression.

`LineNumberNode`s originating from user code are kept, so that the generated code remains
attributable to the user's source lines. This matters for code coverage: Julia only emits a
coverage counter for a line that a `LineNumberNode` points at, so stripping the user's
`LineNumberNode`s leaves every statement of a `@tensor begin ... end` block after the first one
without any coverage information at all. Conversely, a synthesized `LineNumberNode` would
re-attribute all statements that follow it to a line in the parser's own source, so both halves
are needed.
"""
function removeinternallinenumbernodes(ex, userfiles)
    if isexpr(ex, :block)
        # within a block, `LineNumberNode`s are statement markers: drop the internal ones
        args = Any[
            removeinternallinenumbernodes(e, userfiles) for e in ex.args
                if !_isinternallinenumber(e, userfiles)
        ]
        return Expr(:block, args...)
    elseif isa(ex, Expr)
        # elsewhere a `LineNumberNode` may be structurally required -- most notably as the
        # mandatory 2nd argument of a `:macrocall` -- so keep all positions here and only
        # recurse into nested blocks
        return Expr(
            ex.head, Any[removeinternallinenumbernodes(e, userfiles) for e in ex.args]...
        )
    else
        return ex
    end
end

_isinternallinenumber(@nospecialize(x), userfiles) =
    x isa LineNumberNode && x.file ∉ userfiles

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
