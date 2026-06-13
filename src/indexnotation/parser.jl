mutable struct TensorParser
    preprocessors::Vector{Any} # any preprocessing steps
    contractiontreebuilder::Any # determine a contraction tree for a contraction involving multiple tensors
    contractiontreesorter::Any # transforms the contraction expression into an expression of nested binary contractions using the tree output from the contractiontreebuilder
    contractioncostcheck::Any

    postprocessors::Vector{Any}
    function TensorParser()
        preprocessors = [
            normalizeindices, expandconj, nconindexcompletion, extracttensorobjects,
        ]
        contractiontreebuilder = defaulttreebuilder
        contractiontreesorter = defaulttreesorter
        contractioncostcheck = nothing
        postprocessors = [_flatten, addtensoroperations]
        return new(
            preprocessors,
            contractiontreebuilder, contractiontreesorter, contractioncostcheck,
            postprocessors
        )
    end
end

function (parser::TensorParser)(ex::Expr)
    verifytensorexpr(ex)
    for p in parser.preprocessors
        ex = p(ex)::Expr
    end
    treebuilder = parser.contractiontreebuilder
    treesorter = parser.contractiontreesorter
    costcheck = parser.contractioncostcheck
    ex = processcontractions(ex, treebuilder, treesorter, costcheck)::Expr
    ex = tensorify(ex)::Expr
    for p in parser.postprocessors
        ex = p(ex)::Expr
    end
    ex = removeinternallinenumbernodes(ex)::Expr
    return ex
end

"""
    verifytensorexpr(ex)

Check that `ex` is a valid tensor expression and throw an `ArgumentError` if not.
Valid tensor expressions satisfy one of the following (recursive) rules):
- The expression is a scalar expression or a tensor expression.
- The expression is an assignment or a definition, and the left hand side and right hand side are valid tensor expressions or scalars.
- The expression is a block, and all subexpressions are valid tensor expressions or scalars.

See also [`istensorexpr`](@ref) and [`isscalarexpr`](@ref).
"""
function verifytensorexpr(ex)
    if isexpr(ex, :block)
        foreach(verifytensorexpr, ex.args)
    elseif isexpr(ex, :macrocall) && ex.args[1] == Symbol("@notensor")
        return
    elseif isassignment(ex) || isdefinition(ex)
        lhs, rhs = getlhs(ex), getrhs(ex)
        if istensor(lhs)
            istensorexpr(rhs) ||
                throw(ArgumentError("@tensor: the following right hand side is not (no longer) recognized as a tensor expression:\n $rhs"))
        else
            (istensorexpr(rhs) && isempty(getindices(rhs))) || isscalarexpr(rhs) ||
                throw(ArgumentError("@tensor: the following right hand side is not (no longer) recognized as a scalar expression:\n $rhs"))
        end
    elseif isa(ex, Expr)
        (istensorexpr(ex)) || # do not test for empty indices; this will throw IndexError later
            isscalarexpr(ex) ||
            throw(ArgumentError("@tensor: the following expression is not (no longer) recognized:\n$ex"))
    end
    return nothing
end

"""
    tensorify(ex)

Main parsing step to transform a tensor expression `ex` into a series of function calls associated
with the primitive building blocks (tensor operations and allocations).
"""
function tensorify(ex::Expr)
    if isexpr(ex, :macrocall) && ex.args[1] == Symbol("@notensor")
        return ex.args[3]
    end
    # assignment case
    if isassignment(ex) || isdefinition(ex)
        lhs, rhs = getlhs(ex), getrhs(ex)
        # exception handling
        if isexpr(rhs, :call) && rhs.args[1] == :throw
            return rhs
        end

        # process left hand side
        if istensor(lhs) && istensorexpr(rhs)
            indices = getindices(rhs)
            if hastraceindices(lhs)
                err = "left hand side of an assignment should have unique indices: $lhs"
                return :(throw(IndexError($err)))
            end
            dst, leftind, rightind = decomposetensor(lhs)
            if Set(vcat(leftind, rightind)) != Set(indices)
                err = "non-matching indices between left and right hand side: $ex"
                return :(throw(IndexError($err)))
            end
            if isassignment(ex)
                if ex.head == :(=)
                    return instantiate(
                        dst, Zero(), rhs, One(), leftind, rightind, ExistingTensor
                    )
                elseif ex.head == :(+=)
                    return instantiate(
                        dst, One(), rhs, One(), leftind, rightind, ExistingTensor
                    )
                else
                    return instantiate(
                        dst, One(), rhs, -One(), leftind, rightind, ExistingTensor
                    )
                end
            else
                # deal with the case that dst can be an existing variable while also a new value is assigned to it
                if dst ∈ getinputtensorobjects(rhs)
                    dst2 = gensym(dst)
                    return quote
                        $dst2 = $dst
                        $(instantiate(dst2, Zero(), rhs, One(), leftind, rightind, NewTensor))
                        $dst = $dst2
                    end
                else
                    return instantiate(
                        dst, Zero(), rhs, One(), leftind, rightind, NewTensor
                    )
                end
            end
        elseif isassignment(ex) && isscalarexpr(lhs)
            if istensorexpr(rhs) && isempty(getindices(rhs))
                tempvar = gensym(string(lhs))
                returnvar = gensym()
                scalar_expr = quote
                    $(instantiate(tempvar, Zero(), rhs, One(), [], [], TemporaryTensor))
                    $returnvar = tensorscalar($tempvar)
                    tensorfree!($tempvar)
                    $returnvar
                end
                return Expr(ex.head, instantiate_scalar(lhs), scalar_expr)
            elseif isscalarexpr(rhs)
                return Expr(ex.head, instantiate_scalar(lhs), instantiate_scalar(rhs))
            end
        else
            return ex # likely an error
        end
    end
    if ex.head == :block # @tensor begin ... end
        return Expr(ex.head, map(tensorify, ex.args)...)
    end
    # constructions of the form: a = @tensor ...
    if isscalarexpr(ex)
        return instantiate_scalar(ex)
    end
    if istensorexpr(ex)
        if !isempty(getindices(ex))
            err = "cannot evaluate $ex to a scalar: uncontracted indices"
            return :(throw(IndexError($err)))
        end
        tempvar = gensym()
        returnvar = gensym()
        return quote
            $(instantiate(tempvar, Zero(), ex, One(), [], [], TemporaryTensor))
            $returnvar = tensorscalar($tempvar)
            tensorfree!($tempvar)
            $returnvar
        end
    end
    return error("invalid syntax in @tensor macro: $ex")
end
tensorify(ex) = ex
