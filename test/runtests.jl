
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using SHIPs
using Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools, Test
using JuLIP, JuLIP.Testing
using JuLIP.Potentials: evaluate!, evaluate_d!, evaluate, evaluate_d
using SHIPs: PolyTransform, PolyCutoff1s, PolyCutoff2s, eval_basis

Deg = TotalDegree(8, 2)
bo = 4
cg = SHIPs.SphericalHarmonics.ClebschGordan(SHIPs.maxL(Deg))
allKL, Nu = SHIPs.generate_KL_tuples(Deg, bo, cg; filter=false)



🚢 = SHIPBasis(Deg, 4, PolyTransform(2, 1.0), PolyCutoff2s(2, 0.2, 2.5))


function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ]

R = randR(20)
B = eval_basis(🚢, R)



allKL
# @testset "SHIPs.jl" begin
#     include("test_jacobi.jl")
#     include("test_transforms.jl")
#     include("test_ylm.jl")
#     include("test_cg.jl")
#     include("test_Bcoeffs.jl")
#     include("test_basis.jl")
#     # include("test_fast.jl")
#     # include("test_pair.jl")
# end
