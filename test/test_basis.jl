
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs, JuLIP, BenchmarkTools

trans = PolyTransform(2, 1.0)
ship = SHIPBasis(3, 15, 2.0, trans, 2, 0.5, 3.0)

length(ship.Nu)

Rs = 1.0 .+ rand(JVecF, 100)
@btime SHIPs.precompute_A!($ship, $Rs)
SHIPs.length_B(ship)
length(ship.A)

ν = ship.Nu[456]
ship.KL[ν]
kk, ll, mrange =  SHIPs._klm(ν, ship.KL)
kk
ll

function runn(Nu, KL, N)
   for n = 1:N
      ν = Nu[n]
      kk, ll, mrange = SHIPs._klm(ν, KL)
   end
end

@btime runn($(ship.Nu), $(ship.KL), 1000)
