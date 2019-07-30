
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using SHIPs, BenchmarkTools
using SHIPs.JacobiPolys: Jacobi
using SHIPs: eval_basis




x = 2*rand() - 1
α, β = rand(), rand()
N = 30
P = zeros(N+1)
dP = zeros(N+1)
J = Jacobi(α, β, N)


SHIPs.JacobiPolys.eval_basis!(P, nothing, J, x)
SHIPs.JacobiPolys.eval_basis_d!(P, dP, nothing, J, x)
@info("Timing for eval_basis!")
@btime SHIPs.JacobiPolys.eval_basis!($P, nothing, $J, $x)
@btime SHIPs.JacobiPolys.eval_basis!($P, nothing, $J, $x)
@info("Timing for eval_basis_d!")
@btime SHIPs.JacobiPolys.eval_basis_d!($P, $dP, nothing, $J, $x)
@btime SHIPs.JacobiPolys.eval_basis_d!($P, $dP, nothing, $J, $x)

# # Julia Bug? => not a bug but a current limitation
# @info("Looking at that strange allocation?")
# using BenchmarkTools
# f(A, B) = A, B
# g(A, B) = A
# A = rand(5)
# B = rand(5)
# @btime f($A, $B);
# @btime f(1, 2);
# @btime g($A, $B);
#
# function runn(P, dP, J, x, N)
#    for n = 1:1000
#       SHIPs.JacobiPolys.eval_basis_d!(P, dP, J, rand(), N)
#    end
#    return nothing
# end
#
# @info("Try again inside a function - no allocation!")
# @btime runn($P, $dP, $J, $x, 15)
