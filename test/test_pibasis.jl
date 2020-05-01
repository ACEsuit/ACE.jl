
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

basis = SHIPs.PIBasis(P1, 2, D, maxdeg)

Nat = 15
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, :X)
evaluate(basis, Rs, Zs, z0)
length(basis, z0)

##

end
