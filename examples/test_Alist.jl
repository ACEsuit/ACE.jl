
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using Test
using SHIPs, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra, SHIPs.JacobiPolys,
      BenchmarkTools
using SHIPs: TransformedJacobi, transform, transform_d, eval_basis!,
             alloc_B, alloc_temp, alloc_temp_d, alloc_dB, IntS,
             eval_basis_d!


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
shpB = SHIPBasis(spec, trans, fcut; filter=false)
shpB2 = SHIPBasis(spec, trans, fcut)
# shpB2 = SHIPBasis(shpB)

##

Nr = 50
Rs, Zs = randR(Nr)
tmp = alloc_temp(shpB, Nr)
B = SHIPs.alloc_B(shpB)
tmp2 = alloc_temp(shpB2, Nr)
B2 = SHIPs.alloc_B(shpB2)

SHIPs.eval_basis!(B, tmp, shpB, Rs, Zs, 0)
SHIPs.eval_basis!(B2, tmp2, shpB2, Rs, Zs, 0)
@show B ≈ B2

@btime SHIPs.eval_basis!($B, $tmp, $shpB, $Rs, $Zs, 0);
@btime SHIPs.eval_basis!($B2, $tmp2, $shpB2, $Rs, $Zs, 0);

##

tmpd = alloc_temp_d(shpB, Nr)
dB = alloc_dB(shpB, Nr)
eval_basis_d!(B, dB, tmpd, shpB, Rs, Zs, 0)


tmpd2 = alloc_temp_d(shpB2, Nr)
dB2 = alloc_dB(shpB2, Nr)
eval_basis_d!(dB2, tmpd2, shpB2, Rs, Zs, 0)

@show dB2 ≈ dB

@btime eval_basis_d!($B, $dB, $tmpd, $shpB, $Rs, $Zs, 0)
@btime eval_basis_d!($dB2, $tmpd2, $shpB2, $Rs, $Zs, 0)

##


# ##
#
# f = let dB2 = dB2, tmpd2 = tmpd2, shpB2 = shpB2, Rs = Rs, Zs = Zs
#    () -> eval_basis_d!(dB2, tmpd2, shpB2, Rs, Zs, 0)
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
# # debugging -> shows that SHIPs.grad_AA_Rj! is correct...
#
# _AA(Rs) = (
#    SHIPs.precompute_A!(tmp2, shpB2, Rs, Zs, 1);
#    SHIPs.precompute_AA!(tmp2, shpB2, 1);
#    return copy(tmp2.AA[1])
#    )
#
# _dAA(Rs, j) = (
#    SHIPs.precompute_dA!(tmpd2, shpB2, Rs, Zs, 1);
#    SHIPs.precompute_AA!(tmpd2, shpB2, 1);
#    SHIPs.grad_AA_Rj!(tmpd2, shpB2, j, Rs, Zs, 1);
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
#    SHIPs.precompute_A!(tmp2, shpB2, Rs, Zs, 1);
#    return copy(tmp2.A[1])
#    )
#
# _dA1(Rs) = begin
#       SHIPs.precompute_dA!(tmpd2, shpB2, Rs, Zs, 1);
#       alist = shpB2.alists[1]
#       dA = zeros(JVec{ComplexF64}, length(alist))
#       for n = 1:length(alist)
#          zklm = alist[n]
#          dA[n] = SHIPs.grad_phi_Rj(Rs[1], 1, zklm, tmpd2)
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
#       SHIPs.precompute_dA!(tmpd2, shpB2, [R], [0], 1);
#       alist = shpB2.alists[1]
#       dphi = zeros(JVec{ComplexF64}, length(alist))
#       for n = 1:length(alist)
#          zklm = alist[n]
#          dphi[n] = SHIPs.grad_phi_Rj(R, 1, zklm, tmpd2)
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
# SH = shpB2.SH
# Y = copy(tmpd2.Y)
# dY = copy(tmpd2.dY)
#
# _Y(R) = SHIPs.eval_basis(SH, R)
# _dY(R, u) = dot.(Ref(u), SHIPs.eval_basis_d(SH, R)[2])
#
# h = 1e-5
# dY0 = _dY(R, u1)
# dYh = (_Y(R+h*u1) - _Y(R)) / h
# norm(imag.(dY0 - dYh), Inf)
# imag.(dY0 - dYh)
