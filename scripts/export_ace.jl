
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using JuLIP, SHIPs, LinearAlgebra

# #---
# #
#
# p = 1
# r0 = 1.0
# pin = 1
# pout = 2
# rin = 0.5
# rout = 3.0
#
# maxdeg = 10
#
# trans = PolyTransform(p, r0)
# Pr = transformed_jacobi(maxdeg, trans, rout, rin; pcut = pout, pin = pin)
#
# # SHIPs.Export.export_ace(@__DIR__() * "/testrbasis.ships", Pr; ntests=5)


#---
V = nothing
N = 1
for maxdeg = 3:8
   basis = SHIPs.Utils.rpi_basis(; species = :Al, N = N, maxdeg=maxdeg)
   global V = SHIPs.Random.randcombine(basis)
   # V.coeffs[1][3] = 0.0
   fname = "/testpot_ord=$(N)_maxn=$(maxdeg)"
   SHIPs.Export.export_ace(@__DIR__() * fname * ".ships", V)
   JuLIP.save_dict(@__DIR__() * fname * ".json", write_dict(V))
   # SHIPs.Export.export_ace_tests(@__DIR__() * "/testpot_$(N)_test", V, 3)

   # export tests with the dimer structure
   # n_atoms = 2
   # # type x y z
   # 0 0.0 0.0 -1.0
   # 0 1.0 2.0 3.0
   at = Atoms(:Al, [ JVecF(0.0, 0.0, -1.0), JVecF(1.0, 2.0, 3.0) ])
   set_pbc!(at, false)
   fptr = open(@__DIR__() * fname * "_dimer_test.dat", "w")
   println(fptr, "E=$(energy(V, at))")
   F = forces(V, at)
   println(fptr, "F[1]=$(F[1])")
   println(fptr, "F[2]=$(F[2])")
   close(fptr)
end

spec1 = V.pibasis.basis1p.spec
spec1[V.pibasis.inner[1].iAA2iA
(V.coeffs[1]/sqrt(4*pi)) |> display

#--- radial basis test

# Pr = basis.pibasis.basis1p.J
# SHIPs.Export.export_ace(@__DIR__() * "/testpot_rbasis.ships", Pr, ntests=5)

#---

for maxdeg = 3:8
   fname = "testpot_ord=$(N)_maxn=$(maxdeg)"
   filelist = [ fname * ".ships",
                fname * ".json",
                fname * "_dimer_test.dat", ]
                # fname * _test_1.dat",
                # "testpot_$(N)_test_2.dat",
                # "testpot_$(N)_test_3.dat",

                # "testpot_$(N)_rbasis.ships" ]
   for f in filelist
      try
         run(`mv ./scripts/$f /Users/ortner/gits/ace-evaluator/test/ships/`)
      catch
      end
   end
end

#---
# export the Si potential



#---
using LinearAlgebra, JuLIP
# tests with the maxdeg=5 potential
fname = "testpot_ord=1_maxn=8"
V = read_dict(load_dict("/Users/ortner/gits/ace-evaluator/test/ships/" * fname * ".json"))
at = Atoms(:Al, [ JVecF(0.0, 0.0, -1.0), JVecF(1.0, 2.0, 3.0) ])
set_pbc!(at, false)
energy(V, at)
# ace : 0.067359063660145199
r = norm(JVecF(0.0, 0.0, -1.0) - JVecF(1.0, 2.0, 3.0))
z = atomic_number(:Al)
Pr = JuLIP.Potentials.evaluate(V.pibasis.basis1p.J, r)
dot(Pr, V.coeffs[1]) * 2 / sqrt(4*pi)
V.coeffs[1] / sqrt(4*pi)

#---
#
# # Pr
#  Array{Float64,1}:
#   0.0014410269433501582
#  -0.008265537347281394
#   0.026973739255694386
#  -0.06603627044910781
#   0.13476005435084246
#  -0.24176900866242013
#
# R(nl=1,0)(r=4.582576)=0.001441
# R(nl=2,0)(r=4.582576)=-0.008266
# R(nl=3,0)(r=4.582576)=0.026974
# R(nl=4,0)(r=4.582576)=-0.066036
# R(nl=5,0)(r=4.582576)=0.134760
# R(nl=6,0)(r=4.582576)=-0.241769
#
# A_r=1(x=0, n=1)=(0.001441)
# A_r=1(x=0, n=2)=(-0.008266)
# A_r=1(x=0, n=3)=(0.026974)
# A_r=1(x=0, n=4)=(-0.066036)
# A_r=1(x=0, n=5)=(0.134760)
# A_r=1(x=0, n=6)=(-0.241769)



#---

# using LinearAlgebra, JuLIP
# # tests with the maxdeg=5 potential
# z = atomic_number(:Al)
# b5 = SHIPs.PIBasisFcn(z, (SHIPs.RPI.PSH1pBasisFcn(5, 0, 0, z),))
# v5 = V.coeffs[1][5]
# fname = "testpot_ord=1_maxn=5"
# V = read_dict(load_dict("/Users/ortner/gits/ace-evaluator/test/ships/" * fname * ".json"))
# V.pibasis.inner[1].AAindices = 1:1
# V.pibasis.inner[1].orders = [1]
# V.pibasis.inner[1].b2iAA = Dict(b5 => 1)
# V.pibasis.inner[1].iAA2iA = 24 * ones(Int, 1,1)
# empty!(V.coeffs[1])
# push!(V.coeffs[1], v5)
#
# at = Atoms(:Al, [ JVecF(0.0, 0.0, -1.0), JVecF(1.0, 2.0, 3.0) ])
# set_pbc!(at, false)
# energy(V, at)
# # ace : 0.067359063660145199
# r = norm(JVecF(0.0, 0.0, -1.0) - JVecF(1.0, 2.0, 3.0))
# z = atomic_number(:Al)
# Pr = JuLIP.Potentials.evaluate(V.pibasis.basis1p.J, r)
# Pr[5] * V.coeffs[1][1] * 2 / sqrt(4*pi)
# fnameonly = @__DIR__() * fname * "_only"
# SHIPs.Export.export_ace(fnameonly * ".ships", V)
# run(`mv $(fnameonly*".ships") /Users/ortner/gits/ace-evaluator/test/ships/`)
# fptr = open(fnameonly * "_dimer_test.dat", "w")
# println(fptr, "E=$(energy(V, at))")
# F = forces(V, at)
# println(fptr, "F[1]=$(F[1])")
# println(fptr, "F[2]=$(F[2])")
# close(fptr)
# run(`mv $(fnameonly * "_dimer_test.dat") /Users/ortner/gits/ace-evaluator/test/ships/`)


# v5 / sqrt(4*pi) ≈ -0.0396764510800229


#---
using Plots
using SHIPs: evaluate
basis = SHIPs.Utils.rpi_basis(; species = :Al, N = 1, maxdeg=6)
Pr = basis.pibasis.basis1p.J
rp = range(Pr.rl, Pr.ru, length=200)
p = plot()
for n = 1:6
   Pn = [ evaluate(Pr, r)[n] for r in rp ]
   plot!(p, rp, Pn, label = "n=$n")
end
display(p)

r = norm(JVecF(0.0, 0.0, -1.0) -  JVecF(1.0, 2.0, 3.0))
