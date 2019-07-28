
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

##


@info("-------- TEST 🚢  Multi-Species-Basis ---------")
using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using SHIPs: eval_basis!, eval_basis, PolyCutoff1s, PolyCutoff2s
using JuLIP.MLIPs: IPSuperBasis
using JuLIP.Testing: print_tf

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end

randR(N) = [ randR() for n=1:N ], zeros(Int16, N)

randR(N, syms) = randR(N)[1], rand( Int16.(atomic_number.(syms)), N )

function randiso()
   K = @SMatrix rand(3,3)
   K = K - K'
   Q = rand([-1,1]) * exp(K)
end

function randiso(Rs, Zs)
   Q = randiso()
   p = shuffle(1:length(Rs))
   return [ Q * R for  R in Rs[p]], Zs[p]
end

##
@info("Testing creation and (de-)dictionisation of a few BasisSpecs")

spec = SparseSHIPBasis(3, :X, 10, 1.5)
println(@test spec == SparseSHIPBasis(3, 10, 1.5))
println(@test decode_dict(Dict(spec)) == spec)

spec = SparseSHIPBasis(3, [:H, :He], 6, 1.5)
println(@test spec == SparseSHIPBasis(3, [1,2], 6, 1.5))
println(@test decode_dict(Dict(spec)) == spec)

spec = SparseSHIPBasis(2, [:H, :He, :Li], 6, 1.5)
println(@test spec == SparseSHIPBasis(2, [1,2,3], 6, 1.5))
println(@test decode_dict(Dict(spec)) == spec)

# ##
# trans = PolyTransform(2, 1.0)
# cutf = PolyCutoff2s(2, 0.5, 3.0)
# ship41 = SHIPBasis(SparseSHIPBasis(3, :X,  11, 1.5), trans, cutf)
# ship42 = SHIPBasis(SparseSHIPBasis(3, [:Si, :C],  11, 1.5), trans, cutf)
# length(ship41), length(ship42)  # -> 587, 11208
#
# len = length.(ship41.NuZ)
# (len[4] + len[1]*len[3] + len[2]*len[2] + len[1]*len[3] + len[4])
#
# length(ship42)
#
# ##
#
# Rs, Zs = randR(20)
# tmp = SHIPs.alloc_temp(ship3)
#
# SHIPs.precompute_A!(tmp, ship3, Rs, Zs)
# B = SHIPs.alloc_B(ship3, Rs)
# eval_basis!(B, tmp, ship32, Rs, Zs, )
#
# ##
#
# Rs, Zs = randR(20, [:Si, :C])
# ship32 = SHIPBasis(SparseSHIPBasis(3, [:Si, :C],  6, 1.5), trans, cutf)
# length(ship32)
# tmp = SHIPs.alloc_temp(ship32)
# SHIPs.precompute_A!(tmp, ship32, Rs, Zs)
# B = SHIPs.alloc_B(ship32, Rs)
# eval_basis!(B, tmp, ship32, Rs, Zs, 6)


trans = PolyTransform(2, 1.0)
cutf = PolyCutoff2s(2, 0.5, 3.0)

ship2 = SHIPBasis(SparseSHIPBasis(2, [1,2], 10, 2.0), trans, cutf)
ship3 = SHIPBasis(SparseSHIPBasis(3, [1,2], 8, 2.0), trans, cutf)
ship4 = SHIPBasis(SparseSHIPBasis(4, [1,2], 6, 1.5), trans, cutf)
ship5 = SHIPBasis(SparseSHIPBasis(5, [1,2],  5, 1.5), trans, cutf)
ships = [ship2, ship3, ship4, ship5]

@info("Test (de-)dictionisation of basis sets")
for ship in ships
   println(@test (decode_dict(Dict(ship)) == ship))
end


@info("Test isometry invariance for 3B-6B 🚢 s")
for ntest = 1:20
   Rs, Zs = randR(20, [:H, :He])
   iz =  rand([1,2])
   BB = [ eval_basis(🚢, Rs, Zs, iz) for 🚢 in ships ]
   RsX, ZsX = randiso(Rs, Zs)
   BBX = [ eval_basis(🚢, RsX, ZsX, iz) for 🚢 in ships ]
   for (B, BX) in zip(BB, BBX)
      print_tf(@test B ≈ BX)
   end
end
println()
