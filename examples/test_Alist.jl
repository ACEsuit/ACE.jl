
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using Test
using SHIPs, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra, SHIPs.JacobiPolys
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
shpB = SHIPBasis( SparseSHIP(Nmax, 10), trans, fcut; filter=false )

##
shpB2 = SHIPBasis2(shpB)


Nr = 50
Rs, Zs = randR(Nr)
tmp = alloc_temp(shpB, Nr)
B = SHIPs.alloc_B(shpB)
tmp2 = alloc_temp(shpB2, Nr)
B2 = SHIPs.alloc_B(shpB2)

using BenchmarkTools

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
