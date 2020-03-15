
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Compress A" begin
##

using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using JuLIP
using JuLIP: evaluate, evaluate_d
using JuLIP.Testing

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ], zeros(Int16, N), 0
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##


@info("Generate a basis")
basis = SHIPs.Utils.TestBasis(4,9)
@show length(basis)
@info("Convert to the SHIP")
ship = SHIP(basis, randcoeffs(basis))
@info("Compress the coefficients")
shipc = SHIPs.compressA(ship)
@info("check the lengths")
length(ship.coeffs[1])
length(shipc.coeffs[1])

# need to check now that the two evaluate the same.
@info("Check correctness")
for n = 1:50
   Rs = SHIPs.rand_vec(ship.J, 10)
   Zs = zeros(Int16, length(Rs))
   z0 = 0
   s1 = evaluate(ship, Rs, Zs, z0)
   s2 = evaluate(shipc, Rs, Zs, z0)
   print_tf(@test s1 ≈ s2)
end
println()


# @info("Evaluation test")
# Rs = SHIPs.rand_vec(ship.J, 30)
# Zs = zeros(Int16, length(Rs))
# z0 = 0
# tmp = SHIPs.alloc_temp(ship, length(Rs))
# tmp_d = SHIPs.alloc_temp_d(ship, length(Rs))
# dEs = zeros(JVecF, 30)
# @info("  evaluate!:")
# @btime SHIPs.evaluate!($tmp, $ship, $Rs, $Zs, $z0)
# @btime SHIPs.evaluate!($tmp, $shipc, $Rs, $Zs, $z0)
# @info("  evaluate_d!:")
# @btime SHIPs.evaluate_d!($dEs, $tmp_d, $ship, $Rs, $Zs, $z0)
# @btime SHIPs.evaluate_d!($dEs, $tmp_d, $shipc, $Rs, $Zs, $z0)

end
