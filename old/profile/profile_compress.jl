
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



##

using ACE, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
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
basis = ACE.Utils.TestBasis(6,13)
@show length(basis)
@info("Convert to the SHIP")
ship = SHIP(basis, randcoeffs(basis))
@info("Compress the coefficients")
shipc = ACE.compressA(ship)
@info("check the lengths")
length(ship.coeffs[1])
length(shipc.coeffs[1])

@info("Evaluation test")
Rs = ACE.rand_vec(ship.J, 30)
Zs = zeros(Int16, length(Rs))
z0 = 0
tmp = ACE.alloc_temp(ship, length(Rs))
tmp_d = ACE.alloc_temp_d(ship, length(Rs))
dEs = zeros(JVecF, 30)
@info("  evaluate!:")
@btime ACE.evaluate!($tmp, $ship, $Rs, $Zs, $z0)
@btime ACE.evaluate!($tmp, $shipc, $Rs, $Zs, $z0)
@info("  evaluate_d!:")
@btime ACE.evaluate_d!($dEs, $tmp_d, $ship, $Rs, $Zs, $z0)
@btime ACE.evaluate_d!($dEs, $tmp_d, $shipc, $Rs, $Zs, $z0)
