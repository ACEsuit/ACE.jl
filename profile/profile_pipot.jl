
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

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
# using BenchmarkTools
# Vgr = SHIPs.GraphPIPot(V)
# tmp = SHIPs.alloc_temp(V, Nat)
# tmpgr = SHIPs.alloc_temp(Vgr, Nat)
# @btime SHIPs.evaluate!($tmp, $V, $Rs, $Zs, $z0)
# @btime SHIPs.evaluate!($tmpgr, $Vgr, $Rs, $Zs, $z0)
# @btime SHIPs.evaluate_old!($tmpgr, $Vgr, $Rs, $Zs, $z0)
#
# #---
#
# @code_warntype SHIPs.evaluate!(tmpgr, Vgr, Rs, Zs, z0)
