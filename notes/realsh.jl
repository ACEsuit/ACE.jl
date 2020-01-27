
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SymPy, StaticArrays

# a basis function is defined by an l and an m tuple:
ll = [1, 3, 3, 4]
mm = [1, -2, -1, 1]
c = rand() + im * rand()

function convert_AA(ll, mm, c)
    # length of the tuple / length of the product
    n = length(ll)

    # create symbols representing the real spherical harmonics
    # by analogy with exp(ikx) = cos(kx) + i sin(kx)
    C = [ symbols("C$j", real=true) for j in 1:length(ll) ]
    S = [ symbols("S$j", real=true) for j in 1:length(ll) ]

    # now express the complex spherical harmonics in terms of the real ones
    # (note we don't need to refer to the YR list since it stores the same
    #  symbols as the C, S lists)
    YR = Any[nothing for _=1:n]
    for j in 1:n
        if mm[j] == 0
            YR[j] = C[j]
        elseif mm[j] > 0
            YR[j] = (1/SymPy.sqrt(2)) * (C[j] - im * S[j])
        else
            YR[j] = ((-1)^mm[j]/SymPy.sqrt(2)) * (C[j] + im * S[j])
        end
    end

    # finally create the symbol for the prefactor
    a = symbols("a", real=true)
    b = symbols("b", real=true)
    coeff = a + im * b

    # let Sympy multiply and simplify the expression
    expr = simplify(real(coeff * prod(YR)))
    println(expr)

    # next, we need to extract the prefactors
    CS = [S; C]
    for ii in CartesianIndices(ntuple(_ -> length(CS), n))
        if length(unique(ii.I)) != length(ii); continue; end
        if sort([ii.I...]) != [ii.I...]; continue; end
        term = prod(CS[SVector(ii.I...)])
        pref = (expr.coeff(term)).subs(a, real(c)).subs(b, imag(c))
        if pref == 0; continue; end
        println(term, " -> ", pref)
    end
    nothing
end


convert_AA(ll, mm, c)
