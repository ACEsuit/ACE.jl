
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------

@info("Basic test of PIPotential construction and evaluation")
maxdeg = 10
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = ACE.SparsePSHDegree()
P1 = ACE.BasicPSH1pBasis(Pr; species = :X, D = D)
basis = ACE.PIBasis(P1, 2, D, maxdeg)
c = ACE.Random.randcoeffs(basis)
V = combine(basis, c)
Nat = 15
Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, :X)
val_basis = real(sum(c .* evaluate(basis, Rs, Zs, z0)))
val_V = evaluate(V, Rs, Zs, z0)
println(@test(val_basis ≈ val_V))
J = evaluate_d(basis, Rs, Zs, z0)
grad_basis = real(sum(c[i] * J[i,:] for i = 1:length(c)))[:]
grad_V = evaluate_d(V, Rs, Zs, z0)
println(@test(grad_basis ≈ grad_V))

#---
# using BenchmarkTools
# Vgr = ACE.GraphPIPot(V)
# tmp = ACE.alloc_temp(V, Nat)
# tmpgr = ACE.alloc_temp(Vgr, Nat)
# @btime ACE.evaluate!($tmp, $V, $Rs, $Zs, $z0)
# @btime ACE.evaluate!($tmpgr, $Vgr, $Rs, $Zs, $z0)
# @btime ACE.evaluate_old!($tmpgr, $Vgr, $Rs, $Zs, $z0)
#
# #---
#
# @code_warntype ACE.evaluate!(tmpgr, Vgr, Rs, Zs, z0)
