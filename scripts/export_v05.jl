
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


##
using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using JuLIP
using JuLIP: evaluate, evaluate_d
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
B3 = SHIPBasis(SparseSHIP(3,  8, wL=2.0), trans, fcut)
coeffs = randcoeffs(B3)
ship = SHIP(B3, coeffs)

rtests = []
for ntest = 1:5
   r = norm(randR())
   Pr = evaluate(B3.J, r)
   push!(rtests, Dict("r" => r, "Pr" => Pr))
end

tests = []
for ntest = 1:5
   Rs, Zs, z0 = randR(12)
   val = evaluate(ship, Rs, Zs, z0)
   push!(tests, Dict("Rs" => Rs, "Zs" => Zs, "z0" => z0, "val" => val))
end
D = Dict(ship)
D["tests"] = tests
D["rtests"] = rtests
JuLIP.save_dict(@__DIR__() * "/ship_v5.json", D)
