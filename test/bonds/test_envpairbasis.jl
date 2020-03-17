
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



@testset "Environment-Bond-Basis" begin

@info("Testset Environment-Bond-Basis")

##

using StaticArrays, Test, Random
using LinearAlgebra
using JuLIP.Testing: print_tf
using JuLIP: alloc_temp

using SHIPs
using JuLIP: evaluate!, evaluate, evaluate_d!
using SHIPs: alloc_B, alloc_dB
using SHIPs.Bonds: envpairbasis

##

function randR(rin, rout)
   R = @SVector randn(3)
   R /= norm(R) * (rin +  (rout-rin) * rand())
end

function rand_env(Nneigs, rnn)
   R0 = randR(0.8*rnn, 1.2*rnn)
   Renv = [ ((0.66*R0) +  randR(0.6*rnn, 2.0 * rnn)) for _=1:Nneigs ]
   return  R0, Renv
end

function randiso(R0, Renv)
   t = 0.0 # rand() * 2*pi
   σ = rand([-1,1])
   σ = -1
   Q =  [ cos(t) sin(t) 0; -sin(t) cos(t) 0; 0 0 σ ]
   v = (R0/norm(R0) - [0,0,1])
   v /= norm(v)
   H = I - 2 * v * v'
   Rot = H * Q * H
   iso = R -> Rot * (R - R0/2) + R0/2
   @assert Rot' * Rot ≈ I
   @assert (norm(iso(R0)) < 1e-12)
   @assert iso.(iso.(Renv)) ≈ Renv
   Renv_ = shuffle(iso.(Renv))
   return Renv_
end

   # r = [ norm(R - dot(R, R0)/norm(R0)^2 * R0) for R in Renv ]
   # r_ = [ norm(R - dot(R, R0)/norm(R0)^2 * R0) for R in Renv_ ]
   # @assert sort(r) ≈ sort(r_)


##

@info("Basic setup and evaluation test")
r0 = 1.0
Benv = envpairbasis(:X, 3; rnn = r0, rcut0 = 2.0, degree = 5, wenv = 1
   )
@show length(Benv)
tmp = alloc_temp(Benv)
R0, Renv = rand_env(10, r0)
B = alloc_B(Benv)
B1 = evaluate!(B, tmp, Benv, R0, Renv)
B2 = evaluate(Benv, R0, Renv)
println(@test B1 ≈ B2)

##

Renv_ = randiso(R0, Renv)
norm(B1, Inf)
@show norm(evaluate(Benv, R0, Renv_) - B1, Inf) / norm(B1, Inf)

# @info("Rotation-invariance test")
# for ntest = 1:30
#    Renv_ = randiso(R0, Renv)
#    # print_tf(@test evaluate(Benv, R0, Renv_) ≈ B1)
#    @show norm(evaluate(Benv, R0, Renv_) - B1, Inf)
# end


##


end
