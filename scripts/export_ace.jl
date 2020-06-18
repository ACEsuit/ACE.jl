
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using JuLIP, SHIPs

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
for maxdeg = 3:6
   basis = SHIPs.Utils.rpi_basis(; species = :Al, N = N, maxdeg=maxdeg)
   global V = SHIPs.Random.randcombine(basis)
   # V.coeffs[1][3] = 0.0
   fname = "/testpot_ord=$(N)_maxn=$(maxdeg)"
   SHIPs.Export.export_ace(@__DIR__() * fname * ".ships", V)
   # SHIPs.Export.export_ace_tests(@__DIR__() * "/testpot_$(N)_test", V, 3)

   # export tests with for the dimer structure
   # n_atoms = 2
   # # type x y z
   # 0 0.0 0.0 -1.0
   # 0 1.0 2.0 3.0
   at = Atoms(:Al, [ JVecF(0.0, 0.0, -1.0), JVecF(1.0, 2.0, 3.0) ])
   set_pbc!(at, false)
   fptr = open(@__DIR__() * fname * "_dimer_test.dat", "w")
   println(fptr, "E=$(energy(V, at))")
   close(fptr)
end

#--- radial basis test

# Pr = basis.pibasis.basis1p.J
# SHIPs.Export.export_ace(@__DIR__() * "/testpot_rbasis.ships", Pr, ntests=5)

#---

for maxdeg = 3:6
   fname = "testpot_ord=$(N)_maxn=$(maxdeg)"
   filelist = [ fname * ".ships",
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
