
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Real SHIP Implementation" begin

##
using SymPy
using PoSH, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using JuLIP
using JuLIP: evaluate, evaluate_d, evaluate!
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
B2 = SHIPBasis(SparseSHIP(2, 3, wL=1.0), trans, fcut)
B3 = SHIPBasis(SparseSHIP(3, 3, wL=1.0), trans, fcut)
B4 = SHIPBasis(SparseSHIP(4, 3, wL=1.0), trans, fcut)
B5 = SHIPBasis(SparseSHIP(5, 3, wL=1.0), trans, fcut)
BB = [B2, B3, B4, B5]

##
@info("Testing Correctness of the C->R Ship conversion")
for B in BB
   coeffs = randcoeffs(B)
   ship = SHIP(B, coeffs)
   @info("bo = $(bodyorder(B)); converting to RSHIP ...")
   rship = PoSH.convertc2r(ship)

   tmp = PoSH.alloc_temp(ship, 10)
   rtmp = PoSH.alloc_temp(rship, 10)

   for nsamples = 1:5
      Rs, Zs, z0 = randR(10)
      Es = evaluate!(tmp, ship, Rs, Zs, z0)
      rEs = evaluate!(rtmp, rship, Rs, Zs, z0)
      # print_tf(@test Es ≈ rEs)
      @show abs(rEs - Es)
   end
   println()
end

# ##
#
# B = SHIPBasis(SparseSHIP(5,  8, wL=2.0), trans, fcut)
# coeffs = randcoeffs(B)
# ship = SHIP(B, coeffs)
# rship = PoSH.convertc2r(ship)

# ##
#
# B = SHIPBasis(SparseSHIP(6, 12, wL=2.0), trans, fcut)
# coeffs = randcoeffs(B)
# ship = SHIP(B, coeffs)
# rship = RSHIP(ship.J, PoSH.SphericalHarmonics.RSHBasis(ship.SH.maxL), ship.zlist,
#               ship.alists, ship.aalists, real.(ship.coeffs))
# Rs, Zs, z0 = randR(10)
# tmp = PoSH.alloc_temp(ship, length(Rs))
# rtmp = PoSH.alloc_temp(rship, length(Rs))
# @btime evaluate!($tmp, $ship, $Rs, $Zs, $z0)
# @btime evaluate!($rtmp, $rship, $Rs, $Zs, $z0)
# # @btime PoSH.evaluate_new!($rtmp, $rship, $Rs, $Zs, $z0)

##

end
