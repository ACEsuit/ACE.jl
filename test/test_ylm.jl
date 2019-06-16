
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Ylm" begin

using SHIPs.SphericalHarmonics, StaticArrays

function explicit_shs(θ, φ)
   Y00 = 0.5 * sqrt(1/π)
   Y1m1 = 0.5 * sqrt(3/(2*π))*sin(θ)*exp(-im*φ)
   Y10 = 0.5 * sqrt(3/π)*cos(θ)
   Y11 = -0.5 * sqrt(3/(2*π))*sin(θ)*exp(im*φ)
   Y2m2 = 0.25 * sqrt(15/(2*π))*sin(θ)^2*exp(-2*im*φ)
   Y2m1 = 0.5 * sqrt(15/(2*π))*sin(θ)*cos(θ)*exp(-im*φ)
   Y20 = 0.25 * sqrt(5/π)*(3*cos(θ)^2 - 1)
   Y21 = -0.5 * sqrt(15/(2*π))*sin(θ)*cos(θ)*exp(im*φ)
   Y22 = 0.25 * sqrt(15/(2*π))*sin(θ)^2*exp(2*im*φ)
   Y3m3 = 1/8 * exp(-3 * im * φ) * sqrt(35/π) * sin(θ)^3
   Y3m2 = 1/4 * exp(-2 * im * φ) * sqrt(105/(2*π)) * cos(θ) * sin(θ)^2
   Y3m1 = 1/8 * exp(-im * φ) * sqrt(21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
   Y30 = 1/4 * sqrt(7/π) * (-3 * cos(θ) + 5 * cos(θ)^3)
   Y31 = -(1/8) * exp(im * φ) * sqrt(21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
   Y32 = 1/4 * exp(2 * im * φ) * sqrt(105/(2*π)) * cos(θ) * sin(θ)^2
   Y33 = -(1/8) * exp(3 * im * φ) * sqrt(35/π) * sin(θ)^3
   return [Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22,
           Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33]
end

@info("Test 1: check complex spherical harmonics against explicit expressions")
nsamples = 30
for n = 1:nsamples
   θ = rand() * π
   φ = (rand()-0.5) * 2*π
   Y = compute_y(3, cos(θ), φ)
   Yex = explicit_shs(θ, φ)
   print((@test Y ≈ Yex), " ")
end
println()

##
@info("Test 2: check faster implementation matches")
nsamples = 30
L = 30
for n = 1:nsamples
   θ = rand() * π
   φ = (rand()-0.5) * 2*π
   Y = compute_y(L, cos(θ), φ)
   r = 0.1+rand()
   R = SVector(r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ))
   Ynew = cYlm_from_cart(L, R)
   print((@test Y ≈ Ynew), " ")
end
println()

# ##  TODO: there is something wrong here! => test errors due to sqrt(neg numbers)
# @info("Test 3: Old vs New Timing")
# L = 15
# θ = rand() * π
# cos_θ = cos(θ)
# φ = (rand()-0.5) * 2*π
# r = 0.1+rand()
# R = SVector(r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ))
# x, y, z, s = SphericalHarmonics.compute_rxz(R)
# coeff = SphericalHarmonics.compute_coefficients(L)
# P = Vector{Float64}(undef, SphericalHarmonics.sizeP(L))
# SphericalHarmonics.compute_p!(L, z, coeff, P)
# Y = Vector{ComplexF64}(undef, SphericalHarmonics.sizeY(L))
# @info("Old Implementation:")
# @btime compute_y!($L, $cos_θ, $φ, $P, $Y)
# @info("New Implementation:")
# @btime cYlm_from_cart!($Y, $L, $r, $x, $z, $s, $P)
# @info("Experimental Real Spherical Harmonics:")
# Yr = Vector{Float64}(undef, SphericalHarmonics.sizeY(L))
# @btime SphericalHarmonics.rYlm_from_cart!($Y, $L, $r, $x, $z, $s, $P)

end # @testset
