
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using JuLIP, ACE, LinearAlgebra


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
   ACE.Export.export_ace(@__DIR__() * fname * ".ace", V)
   JuLIP.save_dict(@__DIR__() * fname * ".json", write_dict(V))
   # ACE.Export.export_ace_tests(@__DIR__() * "/testpot_$(N)_test", V, 3)
   export_dimer_test(V, fname)
end


#--- copy the first test over to the ace evaluator

for maxdeg = 8:8
   fname = "testpot_ord=$(N)_maxn=$(maxdeg)"
   filelist = [ fname * ".ace",
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
ACE.Export.export_ace(@__DIR__() * fname * ".ace", V)
JuLIP.save_dict(@__DIR__() * fname * ".json", write_dict(V))
ACE.Export.export_ace_tests(@__DIR__() * fname * "_test", V, 1, nrepeat=2)
export_dimer_test(V, fname, :Al)
export_trimer_test(V, fname, :Al)

filelist = [ fname * ".ace",
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

for N in [2, 3, 5], maxdeg in [6, 10]

   rcut = 6.5

   basis = ACE.Utils.rpi_basis(; species = :Al, N = N, maxdeg=maxdeg, rcut=rcut)
   V = ACE.Random.randcombine(basis)
   fname = "/testpot_ord=$(N)_maxn=$(maxdeg)"
   ACE.Export.export_ace(@__DIR__() * fname * ".ace", V)
   JuLIP.save_dict(@__DIR__() * fname * ".json", write_dict(V))
   ACE.Export.export_ace_tests(@__DIR__() * fname * "_test", V, 1)
   export_dimer_test(V, fname, :Al)
   export_trimer_test(V, fname, :Al)

   filelist = [ fname * ".ace",
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
# export the Si potential
