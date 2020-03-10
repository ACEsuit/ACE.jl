
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------




@testset "PureBasis" begin

##
@info("--------- Testing PureBasis ----------")
using LinearAlgebra, Test
using SHIPs
using SHIPs.OrthPolys: OrthPolyBasis, OneOrthogonal
using JuLIP: evaluate, evaluate_d, evaluate!, evaluate_d!

##

@info("Generate many-body 1-orth basis")

deg = 10  # degree
r0, rcut = 0.5, 2.5
trans = PolyTransform(2, r0)
t0, tcut = trans(r0), trans(rcut)
tdf = range(t0, tcut, length=1000)
ww = ones(length(tdf))
J = OrthPolyBasis(deg,  0, trans(r0), 2, trans(rcut), tdf, ww)
# J1 = OneOrthogonal(J)
P = SHIPs.TransformedPolys(J, trans, r0, rcut)

spec = SparseSHIP(2, deg-1; wL = 1.5)
shpB = SHIPBasis(spec, P)
pureB2 = SHIPs.PureBasis(shpB, 2)
pureB1 = SHIPs.PureBasis(shpB, 1)

##

Rs = SHIPs.rand_vec(shpB.J, 2)
Zs = zeros(Int16, 2)
b = evaluate(shpB, Rs, Zs, 0)
bp = pureB2(Rs)
norm(bp)
# @assert length(b) == length(shpB) == length(bp)

##

G1 = let nsamples = 10_000
   G = zeros(length(pureB1), length(pureB1))
   for _ = 1:nsamples
      Rs = SHIPs.rand_vec(shpB.J, 1)
      b = pureB1(Rs)
      G += b * b' / nsamples
   end
   G
end

cond(G1)

##

G2 = let nsamples = 100_000
   G = zeros(length(pureB2), length(pureB2))
   for _ = 1:nsamples
      Rs = SHIPs.rand_vec(shpB.J, 2)
      b = pureB2(Rs)
      G += b * b' / nsamples
   end
   G
end

D = Diagonal( diag(G2).^(-0.5) )
Gsc = D * G2 * D
@show cond(Gsc)           # but not the ∏A basis

## 
end
