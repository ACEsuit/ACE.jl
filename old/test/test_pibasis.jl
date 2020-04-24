
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



@testset "PIBasis" begin

##

@info("-------- TEST PIBASIS ---------")

using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using SHIPs: PolyCutoff1s, PolyCutoff2s
using JuLIP.Testing: print_tf
using Printf
using JuLIP: evaluate!, evaluate, evaluate_d!, evaluate_d

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ], zeros(Int16, N)
randR(N, syms) = randR(N)[1], rand( Int16.(atomic_number.(syms)), N )

##

N = 3
deg = 8

trans = PolyTransform(2, 1.0)
fcut = PolyCutoff2s(2, 0.5, 3.0)
Pr = SHIPs.TransformedJacobi(deg+1, trans, fcut)

basis = SHIPs.PIBasis(N, Pr; totaldegree=deg)
@show length(basis)

Rs, Zs = randR(12)
evaluate(basis, Rs, Zs, 0)

## quick performance check - make sure there are no allocations
B = SHIPs.alloc_B(basis)
tmp = SHIPs.alloc_temp(basis)
@btime evaluate!($B, $tmp, $basis, $Rs, $Zs, 0)
println(@test evaluate(basis, Rs, Zs, 0) == evaluate!(B, tmp, basis, Rs, Zs, 0))

## check permutation-invariance
for ntest = 1:30
   Rs, Zs = randR(12)
   B1 = evaluate(basis, Rs, Zs, 0)
   B2 = evaluate(basis, shuffle(Rs), Zs, 0)
   print_tf(@test B1 ≈ B2)
end

##

end
