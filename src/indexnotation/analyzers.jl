# Tools to analyze parts of tensor expressions and extract information

# analyze tensor expressions and extract information
function decomposetensor(ex)
    if isexpr(ex, [:ref, :typed_hcat])
        if length(ex.args) == 1
            return ex.args[1], Any[], Any[]
        elseif isexpr(ex.args[2], :parameters)
            return ex.args[1], ex.args[3:end], ex.args[2].args
        else
            return ex.args[1], ex.args[2:end], Any[]
        end
    elseif isemptysplittensor(ex)
        return ex.args[1], Any[], Any[]
    elseif isexpr(ex, :typed_vcat)
        if isexpr(ex.args[2], [:row, :tuple])
            leftind = ex.args[2].args
        else
            leftind = ex.args[2:2]
        end
        if isexpr(ex.args[3], [:row, :tuple])
            rightind = ex.args[3].args
        else
            rightind = ex.args[3:3]
        end
        return ex.args[1], leftind, rightind
    end
    throw(ArgumentError("not a valid tensor expression: $ex"))
end
gettensorobject(ex) = decomposetensor(ex)[1]
getleftindices(ex) = decomposetensor(ex)[2]
getrightindices(ex) = decomposetensor(ex)[3]

# """
#     decomposegeneraltensor(ex) -> object, leftind, rightind, α, conj

# Decompose a generalized tensor expression `ex` by extracting information about the tensor
# object, the left and right indices, the scalar factor, and the conjugation flag.
# """
function decomposegeneraltensor(ex)
    if istensor(ex)
        object, leftind, rightind = decomposetensor(ex)
        return (object, leftind, rightind, One(), false)
    elseif isexpr(ex, :call)
        if ex.args[1] == :+ && length(ex.args) == 2 # unary plus: pass on
            return decomposegeneraltensor(ex.args[2])
        elseif ex.args[1] == :- && length(ex.args) == 2 # unary minus: flip scalar factor
            (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[2])
            return (object, leftind, rightind, Expr(:call, :-, α), conj)
        elseif ex.args[1] == :conj && length(ex.args) == 2 # conjugation: flip conjugation flag and conjugate scalar factor
            (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[2])
            return (
                object, leftind, rightind, α === One() ? α : Expr(:call, :conj, α),
                !conj,
            )
        elseif ex.args[1] == :* && length(ex.args) >= 3 &&
                count(a -> isgeneraltensor(a), ex.args) == 1 # scalar multiplication: multiply scalar factors
            idx = findfirst(a -> isgeneraltensor(a), ex.args)
            (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[idx])
            scalar = One()
            for i in 2:length(ex.args)
                scalar = simplify_scalarmul(scalar, i == idx ? α : ex.args[i])
            end
            return (object, leftind, rightind, scalar, conj)
        elseif ex.args[1] == :/ && length(ex.args) == 3 # scalar multiplication: muliply scalar factors
            if isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2])
                (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[2])
                return (object, leftind, rightind, Expr(:call, :/, α, ex.args[3]), conj)
            end
        elseif ex.args[1] == :\ && length(ex.args) == 3 # scalar multiplication: muliply scalar factors
            if isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3])
                (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[3])
                return (object, leftind, rightind, Expr(:call, :\, ex.args[2], α), conj)
            end
        end
    end
    throw(ArgumentError("not a valid generalized tensor expression: $ex"))
end

# get left and right hand side from an expression or definition
function getlhs(ex)
    if isexpr(ex, [:(=), :(+=), :(-=), :(:=), :(≔)])
        return ex.args[1]
    else
        throw(ArgumentError("invalid assignment or definition: $ex"))
    end
end
function getrhs(ex)
    if isexpr(ex, [:(=), :(+=), :(-=), :(:=), :(≔)])
        return ex.args[2]
    else
        throw(ArgumentError("invalid assignment or definition: $ex"))
    end
end

# get all the tensors and tensor objects in a tensor expression (not a definition or assignment)
function gettensors(ex)
    if istensor(ex)
        Any[ex]
    elseif istensorexpr(ex)
        list = Any[]
        for e in ex.args
            append!(list, gettensors(e))
        end
        return list
    else
        return Any[]
    end
end
gettensorobjects(ex) = gettensorobject.(gettensors(ex))

# get all the existing tensor objects which are inputs (i.e. appear in the rhs of assignments and definitions)
function getinputtensorobjects(ex)
    list = Any[]
    if istensorexpr(ex)
        append!(list, gettensorobjects(ex))
    elseif isdefinition(ex)
        append!(list, gettensorobjects(getrhs(ex)))
    elseif isassignment(ex)
        if ex.head == :(+=) || ex.head == :(-=)
            lhs = getlhs(ex)
            if istensor(lhs)
                push!(list, gettensorobject(lhs))
            end
        end
        append!(list, gettensorobjects(getrhs(ex)))
    elseif isexpr(ex, :block)
        for i in 1:length(ex.args)
            list2 = getinputtensorobjects(ex.args[i])
            for j in 1:(i - 1)
                # if objects have previously been defined or assigned to, they are not inputs
                list2 = setdiff(list2, getnewtensorobjects(ex.args[j]))
                list2 = setdiff(list2, getoutputtensorobjects(ex.args[j]))
            end
            append!(list, list2)
        end
    elseif isexpr(ex, :call) && ex.args[1] == :tensorscalar
        append!(list, gettensorobjects(ex.args[2]))
    end
    return unique!(list)
end
# get all the existing tensor objects which are outputs (i.e. appear in the lhs of assignments)
function getoutputtensorobjects(ex)
    list = Any[]
    if isassignment(ex)
        lhs = getlhs(ex)
        if istensor(lhs)
            push!(list, gettensorobject(lhs))
        end
    elseif isexpr(ex, :block)
        for i in 1:length(ex.args)
            list2 = getoutputtensorobjects(ex.args[i])
            for j in 1:(i - 1)
                # if objects have previously been defined, they are not existing outputs
                list2 = setdiff(list2, getnewtensorobjects(ex.args[j]))
            end
            append!(list, list2)
        end
    end
    return unique!(list)
end
# get all the existing tensor objects which are newly created (i.e. appear in the lhs of definition)
function getnewtensorobjects(ex)
    list = Any[]
    if isdefinition(ex)
        lhs = getlhs(ex)
        if istensor(lhs)
            push!(list, gettensorobject(lhs))
        end
    elseif isexpr(ex, :block)
        for e in ex.args
            append!(list, getnewtensorobjects(e))
        end
    end
    return unique!(list)
end

# for any tensor expression, get the list of uncontracted indices that would remain after evaluating that expression
function getindices(ex)
    if istensor(ex)
        _, leftind, rightind = decomposetensor(ex)
        return unique2(vcat(leftind, rightind))
    elseif isexpr(ex, :call)
        if (ex.args[1] == :+ || ex.args[1] == :-)
            return getindices(ex.args[2]) # getindices on any of the args[2:end] should yield the same result
        elseif ex.args[1] == :*
            indices = getindices(ex.args[2])
            for k in 3:length(ex.args)
                append!(indices, getindices(ex.args[k]))
            end
            return unique2(indices)
        elseif ex.args[1] == :/
            return getindices(ex.args[2])
        elseif ex.args[1] == :\
            return getindices(ex.args[3])
        elseif length(ex.args) == 2
            return getindices(ex.args[2])
        end
    end
    return Any[]
end

# get all unique indices appearing in a tensor expression
function getallindices(ex)
    if istensor(ex)
        _, leftind, rightind = decomposetensor(ex)
        return unique!(vcat(leftind, rightind))
    elseif isassignment(ex) || isdefinition(ex)
        return getallindices(getrhs(ex))
    elseif isa(ex, Expr)
        return unique!(mapreduce(getallindices, vcat, ex.args))
    end
    return Any[]
end

function simplify_scalarmul(exa, exb)
    if exa === One()
        return exb
    elseif exb === One()
        return exa
    end
    if isexpr(exa, :call) && exa.args[1] == :* && isexpr(exb, :call) && exb.args[1] == :*
        return Expr(:call, :*, exa.args[2:end]..., exb.args[2:end]...)
    elseif isexpr(exa, :call) && exa.args[1] == *
        return Expr(:call, :*, exa.args[2:end]..., exb)
    elseif isexpr(exb, :call) && exb.args[1] == *
        return Expr(:call, :*, exa, exb.args[2:end]...)
    else
        return Expr(:call, :*, exa, exb)
    end
end
function simplify_scalarmul(exa, exb, exc, exd...)
    return simplify_scalarmul(simplify_scalarmul(exa, exb), exc, exd...)
end

# # auxiliary routine
# function unique2(itr)
#     out = collect(itr)
#     i = 1
#     while i < length(out)
#         inext = findnext(isequal(out[i]), out, i+1)
#         if inext == nothing
#             i += 1
#             continue
#         end
#         while inext !== nothing
#             deleteat!(out, inext)
#             inext = findnext(isequal(out[i]), out, i+1)
#         end
#         deleteat!(out, i)
#     end
#     out
# end
