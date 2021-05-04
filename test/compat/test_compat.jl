
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "TestCompat" begin

#---

using JuLIP, ACE
import ZipFile, JSON

include("../artifacts.jl")
testdatadir = joinpath(datadir, "tests")

#---

@info("Test auto-generated testset `v07_compat`")
zipname = joinpath(testdatadir, "v07_compat.zip")
zipdir = ZipFile.Reader(zipname)

for fptr in zipdir.files
   print("    $(fptr.name): ")
   D = JSON.parse(fptr)
   calc = read_dict(D)
   ACE.Testing.runtests(calc, D["_tests"])
   if fptr.name[1:8] == "rpibasis"
      basis = ACE.Testing.test_basis(D["_params"])
      print(" / ")
      ACE.Testing.runtests(basis, D["_tests"])
   end
   println()
end

close(zipdir)


#---

@info("Test auto-generated testset `v08_compat`")
zipname = joinpath(testdatadir, "v08_compat.zip")
zipdir = ZipFile.Reader(zipname)

for fptr in zipdir.files
   print("    $(fptr.name): ")
   D = JSON.parse(fptr)
   calc = read_dict(D)
   ACE.Testing.runtests(calc, D["_tests"])
   if fptr.name[1:8] == "rpibasis"
      basis = ACE.Testing.test_basis(D["_params"])
      print(" / ")
      ACE.Testing.runtests(basis, D["_tests"])
   end
   println()
end

close(zipdir)


end
