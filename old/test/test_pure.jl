
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------




@testset "PureBasis" begin

##
@info("--------- Testing PureBasis ----------")
using LinearAlgebra, Test
using ACE
using ACE.OrthPolys: OrthPolyBasis, OneOrthogonal
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
P = ACE.TransformedPolys(J, trans, r0, rcut)

spec = SparseSHIP(2, deg-1; wL = 1.5)
aceB = SHIPBasis(spec, P)

pureB = ACE.PureBasis(aceB)
pureB1 = ACE.PureBasis(aceB, 1)
pureB2 = ACE.PureBasis(aceB, 2)

##

Rs = ACE.rand_vec(aceB.J, 2)
Zs = zeros(Int16, 2)
b = evaluate(aceB, Rs, Zs, 0)
bp = pureB2(Rs)
norm(bp)
# @assert length(b) == length(aceB) == length(bp)

##

function condtest(basis, nneigs;
                  nsamples = 10_000,
                  randfun = ()->ACE.rand_vec(aceB.J, nneigs) )
   G = zeros(length(basis), length(basis))
   for _ = 1:nsamples
      Rs = randfun()
      b = basis(Rs)
      G += b * b' / nsamples
   end
   D = Diagonal( diag(G).^(-0.5) )
   Gscal = D * G * D
   return cond(Gscal)
end


##

@info("Conditioning of a pure 1N basis, 1 neig")
@show condtest(pureB1, 1)

@info("Conditioning of a pure 1N basis, 5 neighbours")
@show condtest(pureB1, 5)

@info("Conditioning of a pure 2N basis, 2 neigs")
@show condtest(pureB2, 2; nsamples=100_000)

@info("Conditioning of a pure 2N basis, 5 neigs")
@show condtest(pureB2, 5; nsamples=100_000)

@info("Conditioning of the full basis, 2 neigs")
@show condtest(pureB, 2; nsamples=100_000)

@info("Conditioning of the full basis, 5 neigs")
@show condtest(pureB, 5; nsamples=100_000)


## 

# now we try the same with a 1-orthogonal basis
@info("Construct 1-orthogonal basis")
J1 = OneOrthogonal(J)
P1 = ACE.TransformedPolys(J1, trans, r0, rcut)
spec = SparseSHIP(2, deg-1; wL = 1.5,
                  filterfcn = ν -> ACE.OrthPolys.filter_oneorth(ν, J1))
aceBoo = SHIPBasis(spec, P1)

pureB1oo = ACE.PureBasis(aceBoo, 1)
pureB2oo = ACE.PureBasis(aceBoo, 2)
pureBoo = ACE.PureBasis(aceBoo)

##

@info("Conditioning of One-Orth pure 1N basis, 1 neig")
@show condtest(pureB1oo, 1)

@info("Conditioning of  One-Orth pure 1N basis, 5 neigs")
@show condtest(pureB1oo, 5)

@info("Conditioning of  One-Orth pure 2N basis, 2 neigs")
@show condtest(pureB2oo, 2; nsamples=100_000)

@info("Conditioning of  One-Orth pure 2N basis, 5 neigs")
@show condtest(pureB2oo, 5; nsamples=100_000)

@info("Conditioning of the full One-Orth basis, 2 neigs")
@show condtest(pureBoo, 2; nsamples=100_000)

@info("Conditioning of the full One-Orth basis, 5 neigs")
@show condtest(pureBoo, 5; nsamples=100_000)

##

end
