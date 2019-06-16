
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# - for both 1s and 2s and p = 2, 3, 4
# - consistency of derivatives (finite-difference test)
# - L2 orthogonality

using SHIPs
using SHIPs: PolyTransform, rbasis, eval_basis, eval_basis_d
using SHIPs.JacobiPolys: Jacobi

trans = PolyTransform(1, 1)
B1 = rbasis(10, trans, 2, 3.0)
B2 = rbasis(10, trans, 2, 0.5, 3.0)

# for r in [rand(10); [3.0]]
P, dP = eval_basis_d(B1, rand())
