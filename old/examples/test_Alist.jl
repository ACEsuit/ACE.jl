
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using Test
using ACE, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra, ACE.JacobiPolys,
      BenchmarkTools
using ACE: TransformedJacobi, transform, transform_d,
             alloc_B, alloc_temp, alloc_temp_d, alloc_dB, IntS

using JuLIP: evaluate!, evaluate_d!


function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ], zeros(Int16, N), 0
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

Nmax = 4
rl, ru = 0.5, 3.0
fcut =  PolyCutoff2s(2, rl, ru)
trans = PolyTransform(2, 1.0)
spec = SparseSHIP(Nmax, 10)
aceB = SHIPBasis(spec, trans, fcut)

bgrp = aceB.bgrps[1]


##

Nr = 50
Rs, Zs = randR(Nr)
tmp = alloc_temp(aceB, Nr)
B = ACE.alloc_B(aceB)
tmp2 = alloc_temp(aceB2, Nr)
B2 = ACE.alloc_B(aceB2)

evaluate!(B, tmp, aceB, Rs, Zs, 0)
evaluate!(B2, tmp2, aceB2, Rs, Zs, 0)
@show B ≈ B2

@btime evaluate!($B, $tmp, $aceB, $Rs, $Zs, 0);
@btime evaluate!($B2, $tmp2, $aceB2, $Rs, $Zs, 0);

##

tmpd = alloc_temp_d(aceB, Nr)
dB = alloc_dB(aceB, Nr)
evaluate_d!(B, dB, tmpd, aceB, Rs, Zs, 0)


tmpd2 = alloc_temp_d(aceB2, Nr)
dB2 = alloc_dB(aceB2, Nr)
evaluate_d!(dB2, tmpd2, aceB2, Rs, Zs, 0)

@show dB2 ≈ dB

@btime evaluate_d!($B, $dB, $tmpd, $aceB, $Rs, $Zs, 0)
@btime evaluate_d!($dB2, $tmpd2, $aceB2, $Rs, $Zs, 0)

##


# ##
#
# f = let dB2 = dB2, tmpd2 = tmpd2, aceB2 = aceB2, Rs = Rs, Zs = Zs
#    () -> evaluate_d!(dB2, tmpd2, aceB2, Rs, Zs, 0)
# end
# runn(f, N) = (for n=1:N; f(); end)
# runn(f, 1)
# ##
#
# using Profile
# Profile.clear()
# @profile runn(f, 100)
# Profile.print()
#
# ##
#
#
# # debugging -> shows that ACE.grad_AA_Rj! is correct...
#
# _AA(Rs) = (
#    ACE.precompute_A!(tmp2, aceB2, Rs, Zs, 1);
#    ACE.precompute_AA!(tmp2, aceB2, 1);
#    return copy(tmp2.AA[1])
#    )
#
# _dAA(Rs, j) = (
#    ACE.precompute_dA!(tmpd2, aceB2, Rs, Zs, 1);
#    ACE.precompute_AA!(tmpd2, aceB2, 1);
#    ACE.grad_AA_Rj!(tmpd2, aceB2, j, Rs, Zs, 1);
#    return copy(tmpd2.dAAj[1])
#    )
#
#
# u1 = rand(JVecF); u1 /= norm(u1)
# jj = 4
#
# for p = 2:12
#    h = 0.1^p
#
#    A = _AA(Rs)
#    Rs_h = copy(Rs); Rs_h[jj] += h * u1
#    A_h = _AA(Rs_h)
#    dA_h = (A_h - A) / h
#
#    dAA = _dAA(Rs, jj)
#    dAA_u1 = [ dot(u1, da) for da in dAA ]
#
#    @show norm(real.(dAA_u1 - dA_h), Inf)
# end
#
#









# h = 1e-5
#
# A = _AA(Rs)
# Rs_h = copy(Rs); Rs_h[1] += h * u1
# A_h = _AA(Rs_h)
# dA_h = (A_h - A) / h
#
# dAA = _dAA1(Rs)
# dAA_u1 = [ dot(da, u1) for da in dAA ]
#
# @show norm(real.(dAA_u1 - dA_h), Inf)
#
# @show imag.(dAA_u1 - dA_h)
#
#
#
# _A(Rs) = (
#    ACE.precompute_A!(tmp2, aceB2, Rs, Zs, 1);
#    return copy(tmp2.A[1])
#    )
#
# _dA1(Rs) = begin
#       ACE.precompute_dA!(tmpd2, aceB2, Rs, Zs, 1);
#       alist = aceB2.alists[1]
#       dA = zeros(JVec{ComplexF64}, length(alist))
#       for n = 1:length(alist)
#          zklm = alist[n]
#          dA[n] = ACE.grad_phi_Rj(Rs[1], 1, zklm, tmpd2)
#       end
#       return dA
#    end
#
# _dA1(Rs)
#
# for p = 2:12
#    h = 0.1^p
#
#    A = _A(Rs)
#    Rs_h = copy(Rs); Rs_h[1] += h * u1
#    A_h = _A(Rs_h)
#    dA_h = (A_h - A) / h
#
#    dA = _dA1(Rs)
#    dA_u1 = [ dot(u1, da) for da in dA ]
#
#    @show norm(imag.(dA_u1 - dA_h), Inf)
# end
#
#
# R = Rs[1]
# _phi(R) = _A([R])
# _dphi(R) = begin
#       ACE.precompute_dA!(tmpd2, aceB2, [R], [0], 1);
#       alist = aceB2.alists[1]
#       dphi = zeros(JVec{ComplexF64}, length(alist))
#       for n = 1:length(alist)
#          zklm = alist[n]
#          dphi[n] = ACE.grad_phi_Rj(R, 1, zklm, tmpd2)
#       end
#       return dphi
#    end
#
# for p = 2:12
#    h = 0.1^p
#
#    phi = _phi(R)
#    phi_h = _phi(R + h * u1)
#    dphi_h = (phi_h - phi) / h
#
#    dphi = dot.(Ref(u1), _dphi(R))
#
#    @show norm(imag.(dphi - dphi_h), Inf)
# end
#
# _phi(R)
# dot.(_dphi(R), Ref(u1))
#
#
#
# tmpd.J ≈ tmpd2.JJ
# tmpd.dJ ≈ tmpd2.dJJ
# tmpd.Y ≈ tmpd2.YY
# tmpd.dY ≈ tmpd2.dYY
#
#
# SH = aceB2.SH
# Y = copy(tmpd2.Y)
# dY = copy(tmpd2.dY)
#
# _Y(R) = evaluate(SH, R)
# _dY(R, u) = dot.(Ref(u), evaluate_d(SH, R)[2])
#
# h = 1e-5
# dY0 = _dY(R, u1)
# dYh = (_Y(R+h*u1) - _Y(R)) / h
# norm(imag.(dY0 - dYh), Inf)
# imag.(dY0 - dYh)
