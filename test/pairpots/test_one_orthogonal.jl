
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using Test

@testset "OneOrthogonal" begin

##
@info("--------- Testing OneOrthogonal ----------")

using LinearAlgebra
using SHIPs
using SHIPs.OrthPolys: OrthPolyBasis
using SHIPs.OneOrthogonalModule: OneOrthogonal
using JuLIP: evaluate, evaluate_d, evaluate!, evaluate_d!

##
# Assemble basis
N = 10
Nquad = 1000
dt =  2 / Nquad
tdf = range(-1.0+dt/2, 1.0-dt/2, length=Nquad)
ww = ones(Nquad) / Nquad
P = OrthPolyBasis(N,  0, 1.0, 2, -1.0, tdf,ww)
Q = OneOrthogonal(P)

# Evaluate basis
p = Matrix{Float64}(undef,Nquad,N)
q = Matrix{Float64}(undef,Nquad,N)
for i = 1:Nquad
    evaluate!(@view(p[i,:]), nothing, P, tdf[i])
    evaluate!(@view(q[i,:]), nothing, Q, tdf[i])
end

# Test spanning property
@test rank(p'*q) == N

# Test orthogonality properties
@test q'*Diagonal(ww)*q ≈ I
@test norm(q[:,1:end-1]'*Diagonal(ww)*one.(tdf)) < sqrt(eps())

# Test derivative
ε = sqrt(eps())
q1 = evaluate(Q, -ε/2)
q2 = evaluate(Q, ε/2)
dq = evaluate_d(Q,0.0)
@test (q2.-q1)./ε ≈ dq

end
