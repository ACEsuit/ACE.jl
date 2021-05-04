
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



using ACE, Random
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate!, evaluate_d!,
             alloc_temp, alloc_temp_d
using ACE: alloc_B, alloc_dB
using BenchmarkTools
using Juno

# (:Si, [:C, :O])
# N = 2:2:6
# maxdeg in [7, 12, 17]
degrees = Dict(2 => [7, 12, 17],
               4 => [7, 10, 13],
               6 => [7, 9, 11])
for species in (:Si, ), N = 2:2:6, maxdeg in degrees[N]
   r0 = 2.3
   rcut = 5.0
   trans = PolyTransform(1, r0)
   Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
   P1 = ACE.BasicPSH1pBasis(Pr; species = species)
   D = ACE.SparsePSHDegree()

   basis = ACE.PIBasis(P1, N, D, maxdeg, evaluator = :classic)
   basisdag = ACE.PIBasis(P1, N, D, maxdeg, evaluator = :dag)

   @info("species = $species; N = $N; length = $(length(basis))")

   for Nat in [5, 50]
      Rs, Zs, z0 = ACE.Random.rand_nhd(Nat, basis.basis1p.J,
                                         species)
      tmp = alloc_temp(basis, Rs, Zs, z0)
      tmpd = alloc_temp_d(basis, Rs, Zs, z0)
      tmpdag = alloc_temp(basisdag, Rs, Zs, z0)
      tmpdagd = alloc_temp_d(basisdag, Rs, Zs, z0)
      B = alloc_B(basis, Rs)
      dB = alloc_dB(basis, Rs)
      # ------------------------------
      println("   Nat = $Nat")
      print("   classic eval : ");
      @btime evaluate!($B, $tmp, $basis, $Rs, $Zs, $z0)
      print("     graph eval : ");
      @btime evaluate!($B, $tmpdag, $basisdag, $Rs, $Zs, $z0)
      print("   classic grad : ");
      @btime evaluate_d!($B, $dB, $tmpd, $basis, $Rs, $Zs, $z0)
      print("     graph grad : ")
      @btime evaluate_d!($B, $dB, $tmpdagd, $basisdag, $Rs, $Zs, $z0)
      println()
   end
end


# species = :Si
# N  = 4
# maxdeg = 7
# r0 = 2.3
# rcut = 5.0
# trans = PolyTransform(1, r0)
# Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
# P1 = ACE.BasicPSH1pBasis(Pr; species = species)
# D = ACE.SparsePSHDegree()
#
# Rs, Zs, z0 = ACE.Random.rand_nhd(30, Pr, species)
# basis = ACE.PIBasis(P1, N, D, maxdeg, evaluator = :classic)
# basisdag = ACE.PIBasis(P1, N, D, maxdeg, evaluator = :dag)
# tmp = alloc_temp(basis, Rs, Zs, z0)
# tmpd = alloc_temp_d(basis, Rs, Zs, z0)
# tmpdag = alloc_temp(basisdag, Rs, Zs, z0)
# tmpdagd = alloc_temp_d(basisdag, Rs, Zs, z0)
# B = alloc_B(basis, Rs)
# dB = alloc_dB(basis, Rs)

# @btime evaluate_d!($B, $dB, $tmpdagd, $basisdag, $Rs, $Zs, $z0)
# @btime evaluate_d!($B, $dB, $tmpdagd, $basisdag, $Rs, $Zs, $z0)


#
# function runN(N, f, args...)
#    for n = 1:N
#       f(args...)
#    end
#    return nothing
# end
#
# runN(10, evaluate_d!, B, dB, tmpd, basis, Rs, Zs, z0)
#
# Juno.@profiler runN(10_000, evaluate_d!, B, dB, tmpd, basis, Rs, Zs, z0)
