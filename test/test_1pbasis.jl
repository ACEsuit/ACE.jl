
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "1-Particle Basis"  begin

##

using ACE
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
P1 = ACE.BasicPSH1pBasis(Pr; species = :X)
Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, :X)
evaluate(P1, Rs, Zs, z0)
evaluate_d(P1, Rs[1], Zs[1], z0)

##

for species in (:X, :Si, [:C, :O, :H])
   @info("species = $species")
   Nat = 15
   P1 = ACE.BasicPSH1pBasis(Pr; species = species)
   @info("   test de-serialisation")
   println(@test(all(JuLIP.Testing.test_fio(P1))))

   @info("   test evaluation")
   for ntest = 1:10
      Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
      A = evaluate(P1, Rs, Zs, z0)
      A_ = sum( evaluate(P1, R, Z, z0) for (R, Z) in zip(Rs, Zs) )
      print_tf(@test A ≈ A_)
   end
   println()
   # test that the specification is reproduced correctly!
   @info("    Check specification is correct")
   for iz0 in numz(P1)
      z0 = i2z(P1, iz0)
      P1_spec = ACE.get_basis_spec(P1, z0)
      P1_spec_2 = [ ACE.get_basis_spec(P1, z0, i) for i = 1:length(P1, z0) ]
      println(@test P1_spec == P1_spec_2)
   end
   # Check gradients
   @info("    Check gradients")
   for ntest = 1:30
      Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
      R, Z = Rs[1], Zs[1]
      U = rand(JVecF) .- 0.5; U /= norm(U)
      A = evaluate(P1, R, Z, z0)
      dA = dot.(Ref(U), evaluate_d(P1, R, Z, z0))
      errs = []
      for p = 2:10
         h = 0.1^p
         Ah = evaluate(P1, R + h * U, Z, z0)
         dAh = (Ah - A) / h
         # @show norm(dA - dAh, Inf)
         push!(errs, norm(dA - dAh, Inf))
      end
      success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
      print_tf(@test success)
   end
   println()
end


##

end
