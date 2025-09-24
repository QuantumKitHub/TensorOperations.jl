contract_op(args...) = +(*(args...), *(args...))
function instantiate_scalartype(ex::Expr)
    if istensor(ex)
        return Expr(:call, :scalartype, gettensorobject(ex))
    elseif isgeneraltensor(ex)
        (object, _, _, α, _) = decomposegeneraltensor(ex)
        return Expr(
            :call, :(Base.promote_op), :*, Expr(:call, :scalartype, object),
            instantiate_scalartype(α)
        )
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-)
        if length(ex.args) > 2
            return Expr(:call, :promote_add, map(instantiate_scalartype, ex.args[2:end])...)
        else
            return instantiate_scalartype(ex.args[2])
        end
    elseif isexpr(ex, :call, 3) && ex.args[1] == :* &&
            istensorexpr(ex.args[2]) && istensorexpr(ex.args[3])
        return Expr(
            :call, :promote_contract, map(instantiate_scalartype, ex.args[2:end])...
        )
    elseif ex.head == :call && ex.args[1] ∈ (:/, :\, :*)
        return Expr(
            :call, :(Base.promote_op), ex.args[1],
            map(instantiate_scalartype, ex.args[2:end])...
        )
    elseif ex.head == :call && ex.args[1] == :conj
        return instantiate_scalartype(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :tensorscalar
        return instantiate_scalartype(ex.args[2])
    elseif isscalarexpr(ex)
        return :(scalartype($ex))
    else
        # return :(eltype($ex)) # would probably lead to doing the same operation twice
        throw(ArgumentError("unable to determine scalartype"))
    end
end
instantiate_scalartype(ex::Number) = typeof(ex)
instantiate_scalartype(ex) = Expr(:call, :scalartype, ex)

function instantiate_scalar(ex::Expr)
    if ex.head == :call && ex.args[1] == :tensorscalar
        @assert length(ex.args) == 2 && istensorexpr(ex.args[2])
        tempvar = gensym()
        returnvar = gensym()
        return quote
            $(instantiate(tempvar, Zero(), ex.args[2], One(), [], [], TemporaryTensor))
            $returnvar = tensorscalar($tempvar)
            tensorfree!($tempvar)
            $returnvar
        end
    elseif ex.head == :call
        return Expr(ex.head, ex.args[1], map(instantiate_scalar, ex.args[2:end])...)
    else
        return Expr(ex.head, map(instantiate_scalar, ex.args)...)
    end
end
instantiate_scalar(ex::Symbol) = ex
instantiate_scalar(ex) = ex

@enum AllocationStrategy ExistingTensor NewTensor TemporaryTensor
function instantiate(
        dst, β, ex::Expr, α, leftind::Vector{Any}, rightind::Vector{Any},
        alloc::AllocationStrategy, scaltype = nothing
    )
    if isgeneraltensor(ex)
        return instantiate_generaltensor(dst, β, ex, α, leftind, rightind, alloc, scaltype)
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-)
        if length(ex.args) == 2
            α′ = (ex.args[1] == :-) ? Expr(:call, :-, α) : α
            return instantiate(dst, β, ex.args[2], α′, leftind, rightind, alloc, scaltype)
        else # linear combination
            return instantiate_linearcombination(
                dst, β, ex, α, leftind, rightind, alloc,
                scaltype
            )
        end
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 &&
            isscalarexpr(ex.args[2]) && istensorexpr(ex.args[3])
        α′ = simplify_scalarmul(α, ex.args[2])
        return instantiate(dst, β, ex.args[3], α′, leftind, rightind, alloc, scaltype)
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 &&
            istensorexpr(ex.args[2]) && isscalarexpr(ex.args[3])
        α′ = simplify_scalarmul(α, ex.args[3])
        return instantiate(dst, β, ex.args[2], α′, leftind, rightind, alloc, scaltype)
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 &&
            istensorexpr(ex.args[2]) && istensorexpr(ex.args[3])
        return instantiate_contraction(dst, β, ex, α, leftind, rightind, alloc, scaltype)
    elseif ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3 &&
            isscalarexpr(ex.args[2]) && istensorexpr(ex.args[3])
        α′ = simplify_scalarmul(α, Expr(:call, :\, ex.args[2], 1))
        return instantiate(dst, β, ex.args[3], α′, leftind, rightind, alloc, scaltype)
    elseif ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3 &&
            istensorexpr(ex.args[2]) && isscalarexpr(ex.args[3])
        α′ = simplify_scalarmul(α, Expr(:call, :/, 1, ex.args[3]))
        return instantiate(dst, β, ex.args[2], α′, leftind, rightind, alloc, scaltype)
    end
    throw(ArgumentError("problem with parsing $ex"))
end

function instantiate_generaltensor(
        dst, β, ex::Expr, α, leftind::Vector{Any}, rightind::Vector{Any},
        alloc::AllocationStrategy, scaltype
    )
    src, srcleftind, srcrightind, α2, conj = decomposegeneraltensor(ex)
    srcind = vcat(srcleftind, srcrightind)
    α = simplify_scalarmul(α, α2)

    p1 = (map(l -> findfirst(isequal(l), srcind), leftind)...,)
    p2 = (map(l -> findfirst(isequal(l), srcind), rightind)...,)
    p = (p1, p2)

    out = Expr(:block)
    if isa(α, Expr)
        αsym = gensym(string(dst) * "_α")
        push!(out.args, Expr(:(=), αsym, instantiate_scalar(α)))
        α = αsym
    end
    if isa(β, Expr)
        βsym = gensym(string(dst) * "_β")
        push!(out.args, Expr(:(=), βsym, instantiate_scalar(β)))
        β = βsym
    end
    if alloc ∈ (NewTensor, TemporaryTensor)
        TCsym = gensym("T_" * string(dst))
        istemporary = Val(alloc === TemporaryTensor)
        TCval = @something(
            scaltype, instantiate_scalartype(α === One() ? src : Expr(:call, :*, α, src))
        )
        push!(out.args, Expr(:(=), TCsym, TCval))
        push!(
            out.args,
            Expr(:(=), dst, :(tensoralloc_add($TCsym, $src, $p, $conj, $istemporary)))
        )
    end

    if hastraceindices(ex)
        traceind = unique(setdiff(setdiff(srcind, leftind), rightind))
        q1 = (map(l -> findfirst(isequal(l), srcind), traceind)...,)
        q2 = (map(l -> findlast(isequal(l), srcind), traceind)...,)
        if any(x -> (x === nothing), (p1..., p2..., q1..., q2...)) ||
                !isperm((p1..., p2..., q1..., q2...)) ||
                length(srcind) != length(leftind) + length(rightind) + 2 * length(traceind)
            err = "trace: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
            return :(throw(IndexError($err)))
        end
        push!(
            out.args, :($dst = tensortrace!($dst, $src, $p, ($q1, $q2), $conj, $α, $β))
        )
        return out
    else
        if any(isnothing, (p1..., p2...)) || !isperm((p1..., p2...)) ||
                length(srcind) != length(leftind) + length(rightind)
            err = "add: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
            return :(throw(IndexError($err)))
        end
        push!(out.args, :($dst = tensoradd!($dst, $src, $p, $conj, $α, $β)))
        return out
    end
end
function instantiate_linearcombination(
        dst, β, ex::Expr, α, leftind::Vector{Any}, rightind::Vector{Any},
        alloc::AllocationStrategy, scaltype
    )
    out = Expr(:block)
    if alloc ∈ (NewTensor, TemporaryTensor)
        scaltype = @something(
            scaltype, instantiate_scalartype(α === One() ? ex : Expr(:call, :*, α, ex))
        )
        push!(
            out.args,
            instantiate(dst, β, ex.args[2], α, leftind, rightind, alloc, scaltype)
        )
    else
        push!(
            out.args,
            instantiate(dst, β, ex.args[2], α, leftind, rightind, alloc, scaltype)
        )
    end
    if ex.args[1] == :- && length(ex.args) == 3
        push!(
            out.args,
            instantiate(
                dst, One(), ex.args[3], Expr(:call, :-, α), leftind, rightind,
                ExistingTensor
            )
        )
    elseif ex.args[1] == :+
        for k in 3:length(ex.args)
            push!(
                out.args,
                instantiate(dst, One(), ex.args[k], α, leftind, rightind, ExistingTensor)
            )
        end
    else
        throw(ArgumentError("unable to instantiate linear combination: $ex"))
    end
    return out
end
function instantiate_contraction(
        dst, β, ex::Expr, α, leftind::Vector{Any}, rightind::Vector{Any},
        alloc::AllocationStrategy, scaltype
    )
    exA = ex.args[2]
    exB = ex.args[3]

    indA = getindices(exA)
    indB = getindices(exB)
    cind = intersect(indA, indB)
    indC = vcat(leftind, rightind)
    oindA = intersect(indA, indC) # in the order they appear in A
    oindB = intersect(indB, indC) # in the order they appear in B

    out = Expr(:block)
    if !isgeneraltensor(exA) || hastraceindices(exA)
        A = gensym(string(dst) * "_A")
        push!(out.args, instantiate(A, Zero(), exA, One(), oindA, cind, TemporaryTensor))
        poA = ((1:length(oindA))...,)
        pcA = length(oindA) .+ ((1:length(cind))...,)
        conjA = false
        Atemp = true
    else
        A, indlA, indrA, αA, conj = decomposegeneraltensor(exA)
        indA = vcat(indlA, indrA)
        poA = (map(l -> findfirst(isequal(l), indA), oindA)...,)
        pcA = (map(l -> findfirst(isequal(l), indA), cind)...,)
        conjA = conj
        α = simplify_scalarmul(α, αA)
        Atemp = false
    end
    if !isgeneraltensor(exB) || hastraceindices(exB)
        B = gensym(string(dst) * "_B")
        push!(out.args, instantiate(B, Zero(), exB, One(), cind, oindB, TemporaryTensor))
        poB = length(cind) .+ ((1:length(oindB))...,)
        pcB = ((1:length(cind))...,)
        conjB = false
        Btemp = true
    else
        B, indlB, indrB, αB, conj = decomposegeneraltensor(exB)
        indB = vcat(indlB, indrB)
        poB = (map(l -> findfirst(isequal(l), indB), oindB)...,)
        pcB = (map(l -> findfirst(isequal(l), indB), cind)...,)
        conjB = conj
        α = simplify_scalarmul(α, αB)
        Btemp = false
    end

    oindAB = vcat(oindA, oindB)
    p1 = (map(l -> findfirst(isequal(l), oindAB), leftind)...,)
    p2 = (map(l -> findfirst(isequal(l), oindAB), rightind)...,)

    pAB = (p1, p2)
    pA = (poA, pcA)
    pB = (pcB, poB)

    if any(x -> (x === nothing), (poA..., pcA..., poB..., pcB..., p1..., p2...)) ||
            !(isperm((poA..., pcA...)) && length(indA) == length(poA) + length(pcA)) ||
            !(isperm((pcB..., poB...)) && length(indB) == length(poB) + length(pcB)) ||
            !(isperm((p1..., p2...)) && length(oindAB) == length(p1) + length(p2))
        err = "contraction: $(tuple(leftind..., rightind...)) from $(tuple(indA...)) and $(tuple(indB...)))"
        return :(throw(IndexError($err)))
    end
    if isa(α, Expr)
        αsym = gensym(string(dst) * "_α")
        push!(out.args, Expr(:(=), αsym, instantiate_scalar(α)))
        α = αsym
    end
    if isa(β, Expr)
        βsym = gensym(string(dst) * "_β")
        push!(out.args, Expr(:(=), βsym, instantiate_scalar(β)))
        β = βsym
    end
    if alloc ∈ (NewTensor, TemporaryTensor)
        TCsym = gensym("T_" * string(dst))
        TCval = @something(
            scaltype,
            begin
                TA = instantiate_scalartype(A)
                TB = instantiate_scalartype(B)
                TAB = :(promote_contract($TA, $TB))
                α === One() ? TAB : :(Base.promote_op(*, $(instantiate_scalartype(α)), $TAB))
            end
        )
        istemporary = Val(alloc === TemporaryTensor)
        initC = Expr(
            :block, Expr(:(=), TCsym, TCval),
            :(
                $dst = tensoralloc_contract(
                    $TCsym, $A, $pA, $conjA, $B, $pB, $conjB, $pAB, $istemporary
                )
            )
        )
        push!(out.args, initC)
    end
    push!(
        out.args,
        :($dst = tensorcontract!($dst, $A, $pA, $conjA, $B, $pB, $conjB, $pAB, $α, $β))
    )
    Atemp && push!(out.args, :(tensorfree!($A)))
    Btemp && push!(out.args, :(tensorfree!($B)))
    (Atemp || Btemp) && push!(out.args, dst)
    return out
end
