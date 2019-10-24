
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


##

@info("-------- TEST FILTERING MECHANISM ---------")
using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using SHIPs: eval_basis!, eval_basis, PolyCutoff1s, PolyCutoff2s
using JuLIP.MLIPs: IPSuperBasis
using JuLIP.Testing: print_tf
using Printf

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
function randiso(Rs)
   Q = randiso()
   return [ Q * R for R in shuffle(Rs) ]
end

##

@info("Construct a 5B 🚢")
trans = PolyTransform(2, 1.0)
cutf = PolyCutoff2s(2, 0.5, 3.0)
ship = SHIPBasis(SparseSHIP(4,  10; wL = 1.5), trans, cutf,
                                    filter=false)

##
@info("Select a basis group and show it doesn't have full rank:")
igrp = findfirst(SHIPs.alllen_bgrp(ship, 1) .== 3)
Igrp = SHIPs.I_bgrp(ship, igrp, 1)

G = zeros(length(Igrp), length(Igrp))
nsamples = 10 * length(Igrp)
Zs = zeros(SHIPs.IntS, 4)
for n = 1:nsamples
   Rs, Zs = randR(4)
   B = SHIPs.eval_basis(ship, Rs, Zs, 0)
   b = B[Igrp]
   global G += b * b' / nsamples
end

println(@test rank(G) < 3)
@show rk = rank(G)

##
@info("Now filter that basis and show that this basis group has reduced to length 1")

fship = SHIPs.filter_rpi_basis(ship, 1_000)
fship = SHIPBasis(SparseSHIP(4,  10; wL = 1.5), trans, cutf;
                  filter=true, Nsamples=100)
println(@test SHIPs.len_bgrp(fship, igrp, 1) == rk)
@show length(ship), length(fship)
