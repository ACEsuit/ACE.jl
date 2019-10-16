
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Ylm" begin

import SHIPs
using JuLIP.Testing
using LinearAlgebra, StaticArrays, BenchmarkTools, Test, Printf
using SHIPs.SphericalHarmonics
using SHIPs.SphericalHarmonics: dspher_to_dcart, PseudoSpherical,
               cart2spher, spher2cart
using SHIPs: eval_basis, eval_basis_d

verbose = false

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

@info("Test: check complex spherical harmonics against explicit expressions")
nsamples = 30
for n = 1:nsamples
   θ = rand() * π
   φ = (rand()-0.5) * 2*π
   r = 0.1+rand()
   R = SVector(r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ))
   SH = SHBasis(3)
   Y = eval_basis(SH, R)
   Yex = explicit_shs(θ, φ)
   print_tf((@test Y ≈ Yex))
end
println()

##
@info("      ... same near pole")
nsamples = 30
for n = 1:nsamples
   θ = rand() * 1e-9
   if θ < 1e-10
      θ = 0.0
   end
   φ = (rand()-0.5) * 2*π
   r = 0.1+rand()
   R = SVector(r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ))
   SH = SHBasis(3)
   Y = eval_basis(SH, R)
   Yex = explicit_shs(θ, φ)
   print_tf((@test Y ≈ Yex || norm(Y - Yes, Inf) < 1e-12))
end
println()
##

verbose=false
@info("Test: check derivatives of associated legendre polynomials")
for nsamples = 1:30
   θ = 0.1+0.4 * pi * rand()
   L = 5
   P = SHIPs.SphericalHarmonics.compute_p(L, θ)
   P1, dP = SHIPs.SphericalHarmonics.compute_dp(L, θ)
   # -------------
   P_eq_P1 = true
   for l = 0:L, m = 0:l
      i = SHIPs.SphericalHarmonics.index_p(l, m)
      if ((m == 0) && !(P[i] ≈ P1[i])) || ((m > 0) && !(P[i] ≈ P1[i] * sin(θ)))
         P_eq_P1 = false; break;
      end
   end
   print_tf(@test P_eq_P1)
   # -------------
   errs = []
   verbose && @printf("     h    | error \n")
   for p = 2:10
      h = 0.1^p
      dPh = (SHIPs.SphericalHarmonics.compute_p(L, θ+h) - P) / h
      push!(errs, norm(dP - dPh, Inf))
      verbose && @printf(" %.2e | %.2e \n", h, errs[end])
   end
   success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()

##

@info("      ... same near pole")
for nsamples = 1:30
   θ = rand() * 1e-8
   L = 5
   P = SHIPs.SphericalHarmonics.compute_p(L, θ)
   _, dP = SHIPs.SphericalHarmonics.compute_dp(L, θ)
   errs = []
   verbose && @printf("     h    | error \n")
   for p = 2:10
      h = 0.1^p
      dPh = (SHIPs.SphericalHarmonics.compute_p(L, θ+h) - P) / h
      push!(errs, norm(dP - dPh, Inf))
      verbose && @printf(" %.2e | %.2e \n", h, errs[end])
   end
   success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()

##

@info("Test : spher-cart conversion")
for nsamples = 1:30
   R = (rand(3) .- 0.5) * (1+rand())
   print_tf((@test R ≈ spher2cart(cart2spher(R))))
end
println()

##

@info("Test : spher-cart jacobian")
φθ(S::PseudoSpherical) = [atan(S.sinφ, S.cosφ), atan(S.sinθ, S.cosθ)]
φθ(R::AbstractVector) = φθ(cart2spher(R))
EE = [ [1,0,0], [0,1,0], [0,0,1] ]
h = 1e-5
for nsamples = 1:30
   R = rand(3)
   S = cart2spher(R)
   dR_dS = [ dspher_to_dcart(S, 1.0/S.sinθ, 0.0) dspher_to_dcart(S, 0.0, 1.0) ]
   dR_dS_h = hcat( (φθ(R+h*EE[1])-φθ(R-h*EE[1])) / (2*h),
                   (φθ(R+h*EE[2])-φθ(R-h*EE[2])) / (2*h),
                   (φθ(R+h*EE[3])-φθ(R-h*EE[3])) / (2*h) )'
   print_tf((@test norm(dR_dS - dR_dS_h, Inf) < 1e-5))
end
println()

##
verbose=true
@info("Test: check derivatives of complex spherical harmonics")
for nsamples = 1:30
   R = @SVector rand(3)
   u = @SVector rand(3); u /= norm(u)
   SH = SHBasis(5)
   Y, dY = eval_basis_d(SH, R)
   dY_u = dot.(Ref(u), dY)
   DY = Matrix(transpose(hcat(dY...)))
   errs = []
   verbose && @printf("     h    | error \n")
   for p = 2:10
      h = 0.1^p
      DYh = similar(DY)
      dYh_u = similar(dY_u)
      Rh = Vector(R)
      for i = 1:3
         Rh[i] += h
         DYh[:, i] = (eval_basis(SH, SVector(Rh...)) - Y) / h
         Rh[i] -= h
         dYh_u += DYh[:, i] * u[i]
      end
      # push!(errs, norm(DY - DYh, Inf))
      push!(errs, norm(dY_u - dYh_u, Inf))
      verbose && @printf(" %.2e | %.2e \n", h, errs[end])
   end
   success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()

##

verbose=true
@info("Test: check derivatives of complex spherical harmonics")
for nsamples = 1:30
   R = @SVector rand(3)
   u = @SVector rand(3); u /= norm(u)
   u = SVector(1.0, 0.0, 0.0)
   SH = SHBasis(5)
   Y, dY = eval_basis_d(SH, R)
   dY_u = dot.(dY, Ref(u))
   errs = []
   verbose && @printf("     h    | error \n")
   for p = 2:10
      h = 0.1^p
      dYh = (eval_basis(SH, R+h*u) - Y) / h
      push!(errs, norm((dYh - dY_u), Inf))
      verbose && @printf(" %.2e | %.2e \n", h, errs[end])
   end
   success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()

##


end # @testset
