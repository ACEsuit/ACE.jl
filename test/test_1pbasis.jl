
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
using JuLIP.Potentials: i2z, numz

##

maxdeg = 8
r0 = 1.0
rcut = 3.0

trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)


Nat = 15
P1 = SHIPs.BasicPSH1pBasis(Pr; species = :X)
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, :X)
evaluate(P1, Rs, Zs, z0)

for species in (:X, :Si, [:C, :O, :H])
   @info("species = $species")
   Nat = 15
   P1 = SHIPs.BasicPSH1pBasis(Pr; species = species)
   for ntest = 1:10
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      A = evaluate(P1, Rs, Zs, z0)
      A_ = sum( evaluate(P1, R, Z, z0) for (R, Z) in zip(Rs, Zs) )
      print_tf(@test A ≈ A_)
   end
   println()
   # test that the specification is reproduced correctly!
   @info("Check specification is correct")
   for iz0 in numz(P1)
      z0 = i2z(P1, iz0)
      P1_spec = SHIPs.get_basis_spec(P1, z0)
      P1_spec_2 = [ SHIPs.get_basis_spec(P1, z0, i) for i = 1:length(P1, z0) ]
      println(@test P1_spec == P1_spec_2)
   end
end
println()


##



##

end
