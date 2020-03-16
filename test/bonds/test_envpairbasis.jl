

@testset "Environment-Bond-Basis" begin

@info("Testset Environment-Bond-Basis")

##

using StaticArrays, Test
using LinearAlgebra
using JuLIP.Testing: print_tf

using SHIPs
using JuLIP: evaluate!, evaluate, evaluate_d!
using SHIPs: alloc_B, alloc_dB
using SHIPs.Bonds: envpairbasis

##

Benv = envpairbasis(:X, 3; rcut0 = 2.0, degree = 5, wenv = 1)


##
