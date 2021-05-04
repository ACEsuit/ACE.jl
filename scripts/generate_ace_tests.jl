
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using JuLIP, ACE, LinearAlgebra
using JuLIP.MLIPs: combine

#--- radial basis test

# Pr = basis.pibasis.basis1p.J
# ACE.Export.export_ace(@__DIR__() * "/testpot_rbasis.ace", Pr, ntests=5)

function export_dimer_test(V, fname, species=:Al)
   # export tests with the dimer structure
   # n_atoms = 2
   # # type x y z
   # 0 0.0 0.0 -1.0
   # 0 1.0 2.0 3.0
   at = Atoms(species, [ JVecF(0.0, 0.0, -1.0), JVecF(1.0, 2.0, 3.0) ])
   set_pbc!(at, false)
   fptr = open(@__DIR__() * fname * "_dimer_test.dat", "w")
   println(fptr, "E=$(energy(V, at))")
   println(fptr, "natoms = 2")
   println(fptr, "# type x y z")
   for n = 1:2
      r = at.X[n]
      println(fptr, "0 $(r[1]) $(r[2]) $(r[3])")
   end
   close(fptr)
end

function export_trimer_test(V, fname, species=:Al)
   # export tests with the trimer structuress
   at = Atoms(species, [ JVecF(0.0, 0.0, -1.0),
                         JVecF(1.0, 2.0, 2.0),
                         JVecF(0.0, 1.0, 1.0)  ])
   set_pbc!(at, false)
   fptr = open(@__DIR__() * fname * "_trimer_test.dat", "w")
   println(fptr, "E=$(energy(V, at))")
   println(fptr, "natoms = 3")
   println(fptr, "# type x y z")
   for n = 1:3
      r = at.X[n]
      println(fptr, "0 $(r[1]) $(r[2]) $(r[3])")
   end
   close(fptr)
end



#--- first test
V = nothing
N = 1
for maxdeg = 8:8
   basis = ACE.Utils.rpi_basis(; species = :Al, N = N, maxdeg=maxdeg)
   global V = ACE.Random.randcombine(basis)
   # V.coeffs[1][3] = 0.0
   fname = "/testpot_ord=$(N)_maxn=$(maxdeg)"
   ACE.Export.export_ace(@__DIR__() * fname * ".acejl", V)
   JuLIP.save_dict(@__DIR__() * fname * ".json", write_dict(V))
   # ACE.Export.export_ace_tests(@__DIR__() * "/testpot_$(N)_test", V, 3)
   export_dimer_test(V, fname)
end


#--- copy the first test over to the ace evaluator

for maxdeg = 8:8
   fname = "testpot_ord=$(N)_maxn=$(maxdeg)"
   filelist = [ fname * ".acejl",
                fname * ".json",
                fname * "_dimer_test.dat", ]
                # fname * _test_1.dat",
                # "testpot_$(N)_test_2.dat",
                # "testpot_$(N)_test_3.dat",

                # "testpot_$(N)_rbasis.ace" ]
   for f in filelist
      try
         run(`mv ./scripts/$f /Users/ortner/gits/ace-evaluator/test/julia/`)
      catch
      end
   end
end


#--- export a 3-body potential with a single 3-b basis function
#    and

ord = 2
rcut = 6.5
maxdeg = 4
basis = ACE.Utils.rpi_basis(; species = :Al, N = ord, maxdeg=maxdeg,
                               D = SparsePSHDegree(wL = 1.0), rcut=rcut)
V = ACE.Random.randcombine(basis)
inner = V.pibasis.inner[1]
for b in keys(inner.b2iAA)
   # if ACE.order(b) == 2
   if all([b1.l == 0 for b1 in b.oneps])
      V.coeffs[1][inner.b2iAA[b]] = 0.0
   end
   # end
end

fname = "/testpot_ord=$(ord)_mini"
ACE.Export.export_ace(@__DIR__() * fname * ".acejl", V)
JuLIP.save_dict(@__DIR__() * fname * ".json", write_dict(V))
ACE.Export.export_ace_tests(@__DIR__() * fname * "_test", V, 1, nrepeat=2)
export_dimer_test(V, fname, :Al)
export_trimer_test(V, fname, :Al)

filelist = [ fname * ".acejl",
             fname * ".json",
             fname * "_dimer_test.dat",
             fname * "_trimer_test.dat",
             fname * "_test_1.dat", ]
for f in filelist
   try
      run(`mv ./scripts/$f /Users/ortner/gits/ace-evaluator/test/julia/`)
   catch
   end
end


#--- export a random many-body potential

for N in [2, 3, 5], maxdeg in [10, ]

   rcut = 6.5

   basis = ACE.Utils.rpi_basis(; species = :Al, N = N, maxdeg=maxdeg, rcut=rcut)
   V = ACE.Random.randcombine(basis)
   fname = "/testpot_ord=$(N)_maxn=$(maxdeg)"
   ACE.Export.export_ace(@__DIR__() * fname * ".acejl", V)
   JuLIP.save_dict(@__DIR__() * fname * ".json", write_dict(V))
   ACE.Export.export_ace_tests(@__DIR__() * fname * "_test", V, 1)
   export_dimer_test(V, fname, :Al)
   export_trimer_test(V, fname, :Al)

   filelist = [ fname * ".acejl",
                fname * ".json",
                fname * "_dimer_test.dat",
                fname * "_trimer_test.dat",
                fname * "_test_1.dat"]
   for f in filelist
      try
         run(`mv ./scripts/$f /Users/ortner/gits/ace-evaluator/test/julia/`)
      catch
      end
   end
end


#---
# a potential with pair potential and repulsive core

# TODO:
#  - add on also E0
#  - move pairpot rcut to 7.0

basis = ACE.Utils.rpi_basis(; species = :Al, N = 4, maxdeg=8, pin = 2)
VN = randcombine(basis; diff=2)
pairbasis = ACE.Utils.pair_basis(; species = :Al, maxdeg = 8, rcut = 5.0)
V2 = combine(pairbasis, rand(8) .* (1:8).^(-2))
V2rep = ACE.PairPotentials.RepulsiveCore(V2, 2.0, -0.1234)
V1 = OneBody(:Al => -1.5 + rand())
V = JuLIP.MLIPs.SumIP(V1, V2rep, VN)

fname = "/ace_reppair"
JuLIP.save_dict(@__DIR__() * fname * ".json", write_dict(V))
# Vl = read_dict(load_dict(@__DIR__() * fname * ".json"))
# V1, V2rep, VN = tuple(Vl.components...)
ACE.Export.export_ace(@__DIR__() * fname * ".acejl", VN, V2rep, V1)
ACE.Export.export_ace_tests(@__DIR__() * fname * "_test", V, 1, s=:Al)
export_dimer_test(V, fname, :Al)
export_trimer_test(V, fname, :Al)

filelist = [ fname * ".acejl",
             fname * ".json",
             fname * "_dimer_test.dat",
             fname * "_trimer_test.dat",
             fname * "_test_1.dat"]
for f in filelist
   try
      run(`mv ./scripts/$f /Users/ortner/gits/ace-evaluator/test/julia/`)
   catch
   end
end




#---
# Cas Si potential

fname = "/Si_B6_N7_18_lap_1.1_rep"
D = load_dict(@__DIR__() * fname * ".json")
Vsi = read_dict(D["IP"])
ACE.Export.export_ace(@__DIR__() * fname * ".acejl",
                      Vsi.components[3], Vsi.components[2], Vsi.components[1])
ACE.Export.export_ace_tests(@__DIR__() * fname * "_test", Vsi, 1, s=:Si)
export_dimer_test(Vsi, fname, :Si)
export_trimer_test(Vsi, fname, :Si)

filelist = [ fname * ".acejl",
             fname * ".json",
             fname * "_dimer_test.dat",
             fname * "_trimer_test.dat",
             fname * "_test_1.dat"]
for f in filelist
   try
      run(`mv ./scripts/$f /Users/ortner/gits/ace-evaluator/test/julia/`)
   catch
   end
end


#---

fname = "/Si_B6_N7_18_lap_1"
VN = Vsi.components[3]
V2 = Vsi.components[2]
V1 = Vsi.components[1]
ACE.Export.export_ace(@__DIR__() * fname * ".acejl", VN)
ACE.Export.export_ace_tests(@__DIR__() * fname * "_test", VN, 1, s=:Si)
export_dimer_test(VN, fname, :Si)
export_trimer_test(VN, fname, :Si)
filelist = [ fname * ".acejl",
             fname * ".json",
             fname * "_dimer_test.dat",
             fname * "_trimer_test.dat",
             fname * "_test_1.dat"]
for f in filelist
   try
      run(`mv ./scripts$f /Users/ortner/gits/ace-evaluator/test/julia/`)
   catch
   end
end

#---
#
# at = Atoms(:Si, [ JVecF(0.0, 0.0, -1.0),
#                       JVecF(1.0, 2.0, 2.0),
#                       JVecF(0.0, 1.0, 1.0)  ])
# set_pbc!(at, false)
#
# energy(VN, at)
#
# site_energies(VN, at)
#
# X = at.X
# [ norm(X[i] - X[j]) for i = 1:2 for j = i+1:3 ]
#
# VN
#
#
# r = 1.732051
# z = AtomicNumber(:Si)
#
# J = VN.pibasis.basis1p.J
# ACE.evaluate(J, r)
# J
