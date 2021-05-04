
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using JuLIP, ACE, ZipFile, JSON

#---

species_list = [ [:Si], [:Al, :Ti] ] #, [:C, :O, :H] ]
degrees = [ 15, 14, 13, 12, 11, 12 ]
wLs = [ 1.5, 1.4, 1.3, 1.6, 1.55, 1.62 ]

zipname = dirname(pathof(ACE))[1:end-3] * "test/models/v08_compat.zip"
zipdir = ZipFile.Writer(zipname)

@info("Exporting several test basis sets and test potentials.")
for species in species_list, N = 1:length(degrees)
   local D, fptr
   @info("   ... species=$(species), N=$(N)");
   r0 = minimum(rnn.(species))
   _params = Dict("species" => species, "maxdeg" => degrees[N], "wL" => wLs[N],
                  "r0" => r0, "rcut" => 2.5 * r0, "N" => N,
                  "degreetype" => "SparsePSHDegree")
   basis = ACE.Testing.test_basis(_params)

   # file label
   flabel = "_$(N)_" * prod(string.(species)) * ".json"

   # write basis and basis tests
   D = write_dict(basis)
   D["_params"] = _params
   D["_tests"] = ACE.Testing.createtests(basis, 3; tests = ["E"])
   fptr = ZipFile.addfile(zipdir, "rpibasis" * flabel; method = ZipFile.Deflate)
   write(fptr, JSON.json(D))

   # write potential and potential tests
   V = ACE.Random.randcombine(basis; diff=1)
   D = write_dict(V)
   D["_tests"] = ACE.Testing.createtests(V, 3)
   fptr = ZipFile.addfile(zipdir, "rpipot" * flabel; method = ZipFile.Deflate)
   write(fptr, JSON.json(D))
end

close(zipdir)
