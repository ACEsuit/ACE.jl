
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

maxdeg = 15
r0 = 1.0
rcut = 3.0

trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)


for species in (:X, :Si, [:C, :O, :H])
   Nat = 15
   P1 = SHIPs.BasicPSH1pBasis(Pr; species = species)
   for ntest = 1:10
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      A = evaluate(P1, Rs, Zs, z0)
      A_ = sum( evaluate(P1, R, Z, z0) for (R, Z) in zip(Rs, Zs) )
      print_tf(@test A ≈ A_)
   end
end

##



##

end
