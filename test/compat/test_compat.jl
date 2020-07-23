
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "TestCompat" begin

#---

using JuLIP, ACE
import ZipFile, JSON

#---

@info("Test auto-generated testset `v07_compat`")
zipname = dirname(pathof(ACE))[1:end-3] * "test/models/v07_compat.zip"
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

end
