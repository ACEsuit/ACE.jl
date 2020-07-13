
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using JuLIP, SHIPs, ZipFile, JSON

#---

species_list = [ [:Si], [:Al, :Ti] ]  # , [:C, :O, :H] ]
degrees = [ 15, 14, 13] # , 12, 11, 12, 13 ]
wLs = [ 1.5, 1.4, 1.3, 1.6, 1.55, 1.45, 1.35 ]

zipname = dirname(pathof(SHIPs))[1:end-3] * "test/models/v07_compat.zip"
zipdir = ZipFile.Writer(zipname)

@info("Exporting several test basis sets and test potentials.")
for species in species_list, N = 1:length(degrees)
   local D, fptr
   @info("   ... species=$(species), N=$(N)");
   r0 = minimum(rnn.(species))
   _params = Dict("species" => species, "maxdeg" => degrees[N], "wL" => wLs[N],
                  "r0" => r0, "rcut" => 2.5 * r0,
                  "degreetype" => "SparsePSHDegree")
   basis = rpi_basis(; species = species, N = N, maxdeg = _params["maxdeg"],
                       r0 = _params["r0"], rcut = _params["rcut"],
                       D = SparsePSHDegree(wL = _params["wL"]) )

   # file label
   flabel = "_$(N)_" * prod(string.(species)) * ".json"

   # write basis and basis tests
   D = write_dict(basis)
   D["_params"] = _params
   D["_tests"] = SHIPs.Testing.createtests(basis, 3; tests = ["E"])
   fptr = ZipFile.addfile(zipdir, "rpibasis" * flabel; method = ZipFile.Deflate)
   write(fptr, JSON.json(D))

   # write potential and potential tests
   V = SHIPs.Random.randcombine(basis; diff=1)
   D = write_dict(V)
   D["_tests"] = SHIPs.Testing.createtests(V, 3)
   fptr = ZipFile.addfile(zipdir, "rpipot" * flabel; method = ZipFile.Deflate)
   write(fptr, JSON.json(D))
end

close(zipdir)
