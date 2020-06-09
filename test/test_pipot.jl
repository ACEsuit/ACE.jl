
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "PIPotential"  begin

#---


using SHIPs
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing, Random
using JuLIP: evaluate, evaluate_d, evaluate_ed
using SHIPs: combine

#---

@info("Basic test of PIPotential construction and evaluation")
maxdeg = 10
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SHIPs.SparsePSHDegree()
P1 = SHIPs.BasicPSH1pBasis(Pr; species = :X, D = D)
basis = SHIPs.PIBasis(P1, 2, D, maxdeg)
c = SHIPs.Random.randcoeffs(basis)
V = combine(basis, c)
Nat = 15
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, :X)
val_basis = real(sum(c .* evaluate(basis, Rs, Zs, z0)))
val_V = evaluate(V, Rs, Zs, z0)
println(@test(val_basis ≈ val_V))
J = evaluate_d(basis, Rs, Zs, z0)
grad_basis = real(sum(c[i] * J[i,:] for i = 1:length(c)))[:]
grad_V = evaluate_d(V, Rs, Zs, z0)
println(@test(grad_basis ≈ grad_V))

#---

# check multi-species
maxdeg = 5
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
species = [:C, :O, :H]
P1 = SHIPs.BasicPSH1pBasis(Pr; species = [:C, :O, :H], D = D)
basis = SHIPs.PIBasis(P1, 3, D, maxdeg)
c = randcoeffs(basis)
V = combine(basis, c)
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
AA = evaluate(basis, Rs, Zs, z0)
val_basis = real(sum(c .* evaluate(basis, Rs, Zs, z0)))
val_V = evaluate(V, Rs, Zs, z0)
println(@test(val_basis ≈ val_V))
J = evaluate_d(basis, Rs, Zs, z0)
grad_basis = real(sum(c[i] * J[i,:] for i = 1:length(c)))[:]
grad_V = evaluate_d(V, Rs, Zs, z0)
println(@test(grad_basis ≈ grad_V))



#---

@info("Check several properties of PIPotential")
for species in (:X, :Si, [:C, :O, :H]), N = 1:5
   maxdeg = 7
   Nat = 15
   P1 = SHIPs.BasicPSH1pBasis(Pr; species = species)
   basis = SHIPs.PIBasis(P1, N, D, maxdeg)
   @info("species = $species; N = $N; length = $(length(basis))")
   c = randcoeffs(basis)
   V = combine(basis, c)
   @info("check (de-)serialisation")
   println(@test(all(JuLIP.Testing.test_fio(V))))
   @info("Check basis and potential match")
   for ntest = 1:20
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      val_basis = real(sum(c .* evaluate(basis, Rs, Zs, z0)))
      val_V = evaluate(V, Rs, Zs, z0)
      print_tf(@test(val_basis ≈ val_V))
   end
   println()
   @info("Check gradients")
   for ntest = 1:20
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      V0 = evaluate(V, Rs, Zs, z0)
      dV0 = evaluate_d(V, Rs, Zs, z0)
      Us = [ rand(eltype(Rs)) .- 0.5 for _=1:length(Rs) ]
      dV0_dUs = sum(transpose.(dV0) .* Us)
      errs = []
      for p = 2:12
         h = 0.1^p
         V_h = evaluate(V, Rs + h * Us, Zs, z0)
         dV_h = (V_h - V0) / h
         # @show norm(dAA_h - dAA_dUs, Inf)
         push!(errs, norm(dV_h - dV0_dUs, Inf))
      end
      success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
      print_tf(@test success)
   end
   println()
   @info("Check graph evaluator")
   Vgr = SHIPs.graph_evaluator(V)
   for ntest = 1:20
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      v = evaluate(V, Rs, Zs, z0)
      vgr = evaluate(Vgr, Rs, Zs, z0)
      print_tf(@test(v ≈ vgr))
   end
   println()
end
println()

#---


@info("Check Correctness of SHIP.PIPotential calculators")

naive_energy(V::PIPotential, at) =
      sum( evaluate(V, Rs, at.Z[j], at.Z[i])
            for (i, j, Rs) in sites(at, cutoff(V)) )

for N = 1:5
   species = :Si
   maxdeg = 7
   Pr = transformed_jacobi(maxdeg, trans, 4.0; pcut = 2)
   P1 = SHIPs.BasicPSH1pBasis(Pr; species = species)
   basis = SHIPs.PIBasis(P1, N, D, maxdeg)
   @info("N = $N; length = $(length(basis))")
   c = randcoeffs(basis)
   V = combine(basis, c)
   at = bulk(:Si) * (2,2,3)
   rattle!(at, 0.1)
   print("     energy: ")
   println(@test energy(V, at) ≈ naive_energy(V, at) )
   print("site-energy: ")
   println(@test energy(V, at) ≈ sum( site_energy(V, at, n)
                                         for n = 1:length(at) ) )
   print("     forces: ")
   println(@test JuLIP.Testing.fdtest(V, at; verbose=false))
   print("site-forces: ")
   println(@test JuLIP.Testing.fdtest( x -> site_energy(V, set_dofs!(at, x), 3),
                                       x -> mat(site_energy_d(V, set_dofs!(at, x), 3))[:],
                                       dofs(at); verbose=false ) )
end


#---

end
