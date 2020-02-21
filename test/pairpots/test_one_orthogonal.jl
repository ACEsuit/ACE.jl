
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "OneOrthogonal" begin

@info("--------- Testing OneOrthogonal ----------")

using PoSH, Test
using PoSH.OrthPolys: OrthPolyBasis
using PoSH.OneOrthogonalModule: OneOrthogonal

# Assemble basis
N = 15
Nquad = 1000
dt =  2 / Nquad
tdf = range(-1.0+dt/2, 1.0-dt/2, length=Nquad)
ww = ones(Nquad)
P = OrthPolyBasis(N,  0, 1.0, 0, -1.0, tdf,ww)
Q = OneOrthogonal(P)

# Evaluate basis
q = Matrix{Float64}(undef,Nquad,N)
for i = 1:Nquad
    evaluate!(@view(q[i,:]), nothing, Q, tdf[i])
end

# Test orthogonality propers
@test q'*Diagonal(ww)*q ≈ I
@test norm(q[:,1:end]'*Diagonal(ww)*one.(tdf)) < sqrt(eps())

# Test derivative
# TODO
