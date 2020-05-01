
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "1-Particle Basis"  begin

##


using SHIPs
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d


##

maxdeg = 10
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SHIPs.SparsePSHDegree()
P1 = SHIPs.BasicPSH1pBasis(Pr; species = :X, D = D)

pib = SHIPs.PermInvariantBasis(P1, 2, D, maxdeg
   )

Nat = 15
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, :X)
evaluate(P1, Rs, Zs, z0)

##

end


Aspec = SHIPs.get_basis_spec(P1, AtomicNumber(0))
length(Aspec)
