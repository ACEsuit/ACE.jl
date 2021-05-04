
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



@testset "Environment-Bond-Basis" begin

@info("Testset Environment-Bond-Basis")

##

using StaticArrays, Test, Random
using LinearAlgebra
using JuLIP.Testing: print_tf
using JuLIP: alloc_temp
using JuLIP.FIO: read_dict, write_dict, load_dict, save_dict

using ACE
using JuLIP: evaluate!, evaluate, evaluate_d!
using ACE: alloc_B, alloc_dB
using ACE.Bonds: envpairbasis

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

"""
This applies several random symmetry operations to the bond configuration
(R0, Renv) :
  - rotate it about the R0 axis
  - reflect along the z axis
  - rotate the entire configuration (R0, Renv) -> (Q R0, Q Renv)
  - random permutatio of Renv
"""
function randiso(R0, Renv)
   r̂ = R0 / norm(R0)
   o = R0 / 2
   t = rand() * 2*pi
   Q =  [ cos(t) sin(t) 0; -sin(t) cos(t) 0; 0 0 1 ]
   v = (r̂ - [0,0,1])
   v /= norm(v)
   H = I - 2 * v * v'
   Rot = H * Q * H
   iso = R -> Rot * (R - o) + o
   isot = R -> Rot' * (R - o) + o
   @assert Rot' * Rot ≈ I
   @assert iso(R0) ≈ R0
   @assert isot.(iso.(Renv)) ≈ Renv
   return randglobalrot(randiso_z(R0, shuffle(iso.(Renv)))...)
end

function randglobalrot(R0, Renv)
   Q = qr(randn(3,3)).Q
   iso = R -> SVector((Q * R)...)
   return iso(R0), iso.(Renv)
end

function randiso_z(R0, Renv)
   σ = rand([0, 1])
   r̂ = R0 / norm(R0)
   iso = R -> SVector( R - σ * 2 * dot(r̂, R-R0/2) * r̂  )
   return R0, shuffle(iso.(Renv))
end

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

@info("Symmetry-invariance test")
for ntest = 1:50
   R0_, Renv_ = randiso(R0, Renv)
   print_tf(@test evaluate(Benv, R0_, Renv_) ≈ B1)
end


##

@info("Generate and test an EnvPairPot")
c = rand(ComplexF64, length(Benv))
V = JuLIP.MLIPs.combine(Benv, c)
tmp = alloc_temp(V)

R0, Renv = rand_env(10, r0)
evaluate!(tmp, V, R0, Renv)

for ntest = 1:50
   R0, Renv = rand_env(10, r0)
   B = evaluate(Benv, R0, Renv)
   print_tf(@test evaluate!(tmp, V, R0, Renv) ≈ real(sum(c .* B)))
end

##

@info("Test (de-)serialisation")
DV = write_dict(V)
V1 = read_dict(DV)
println(@test V1 == V)
tmpname = tempname()
save_dict(tmpname, DV)
V2 = read_dict(load_dict(tmpname))
println(@test V == V2)

##
end
