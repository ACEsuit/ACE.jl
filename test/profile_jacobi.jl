
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using SHIPs, BenchmarkTools
using SHIPs.JacobiPolys: Jacobi, eval_basis, eval_grad





x = 2*rand() - 1
α, β = rand(), rand()
N = 30
P = zeros(N+1)
dP = zeros(N+1)
J = Jacobi(α, β, N)

SHIPs.JacobiPolys.eval_basis!(P, J, x, 15)
SHIPs.JacobiPolys.eval_grad!(P, dP, J, x, 15)
@info("Timing for eval_basis!")
@btime SHIPs.JacobiPolys.eval_basis!($P, $J, $x, 15)
@btime SHIPs.JacobiPolys.eval_basis!($P, $J, $x, 15)
@info("Timing for eval_grad!")
@btime SHIPs.JacobiPolys.eval_grad!($P, $dP, $J, $x, 15)
@btime SHIPs.JacobiPolys.eval_grad!($P, $dP, $J, $x, 15)

# Julia Bug?
@info("Looking at that strange allocation?")
using BenchmarkTools
f(A, B) = A, B
g(A, B) = A
A = rand(5)
B = rand(5)
@btime f($A, $B);
@btime f(1, 2);
@btime g($A, $B);
