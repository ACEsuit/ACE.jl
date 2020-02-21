
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using JuLIP
using JuLIP: evaluate, evaluate_d, evaluate!, evaluate_d!
using JuLIP.Testing

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ], zeros(Int16, N), 0
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

trans = PolyTransform(3, 1.0)
fcut = PolyCutoff2s(2, 0.5, 3.0)
B = SHIPBasis(SparseSHIP(6, 12, wL=2.0), trans, fcut)
coeffs = randcoeffs(B)

# a complex SHIP potential
ship = SHIP(B, coeffs)
# a real SHIP potential with the same length basis
rship = RSHIP(ship.J, SHIPs.SphericalHarmonics.RSHBasis(ship.SH.maxL), ship.zlist,
              ship.alists, ship.aalists, real.(ship.coeffs))
# the honestly converted real SHIP potential
rship1 = SHIPs.convertc2r(ship)
# and then compressed to a minimal basis ...
rship2 = SHIPs.compressA(rship1)

##

# evaluate! benchmark
Rs, Zs, z0 = randR(10)
tmp = SHIPs.alloc_temp(ship, length(Rs))
rtmp = SHIPs.alloc_temp(rship, length(Rs))
rtmp1 = SHIPs.alloc_temp(rship1, length(Rs))
rtmp2 = SHIPs.alloc_temp(rship2, length(Rs))
@info("Profile `evaluate!`")
print("       complex SHIP: "); @btime evaluate!($tmp, $ship, $Rs, $Zs, $z0)
print("RSHIP - same length: "); @btime evaluate!($rtmp, $rship, $Rs, $Zs, $z0)
print("RSHIP -   converted: "); @btime evaluate!($rtmp1, $rship1, $Rs, $Zs, $z0)
print("RSHIP -  compressed: "); @btime evaluate!($rtmp2, $rship2, $Rs, $Zs, $z0)

##

# evaluate_d! benchmark
tmpd = SHIPs.alloc_temp_d(ship, length(Rs))
rtmpd = SHIPs.alloc_temp_d(rship, length(Rs))
rtmp1d = SHIPs.alloc_temp_d(rship1, length(Rs))
rtmp2d = SHIPs.alloc_temp_d(rship2, length(Rs))
dEs = zeros(JVecF, length(Rs))
@info("Profile `evaluate_d!`")
print("       complex SHIP: "); @btime evaluate_d!(dEs, $tmpd, $ship, $Rs, $Zs, $z0)
print("RSHIP - same length: "); @btime evaluate_d!(dEs, $rtmpd, $rship, $Rs, $Zs, $z0)
print("RSHIP -   converted: "); @btime evaluate_d!(dEs, $rtmp1d, $rship1, $Rs, $Zs, $z0)
print("RSHIP -  compressed: "); @btime evaluate_d!(dEs, $rtmp2d, $rship2, $Rs, $Zs, $z0)
