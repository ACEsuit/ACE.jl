
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using ACE, Pkg
using Pkg.Artifacts

artifacts_toml = joinpath(pathof(ACE)[1:end-11], "Artifacts.toml")

# add ACEData@0.8.0 to Artifacts if it doesn't yet exist

label = "acedata_v0.8.0"
tarname = "v0.8.0.tar.gz"
url = "https://github.com/cortner/ACEData/archive/" * tarname

data_hash = artifact_hash(label, artifacts_toml)

if data_hash == nothing || !artifact_exists(data_hash)
    tarfile = download(url, @__DIR__() * "/" * tarname)
    hash_ = create_artifact() do artifact_dir
        cp(tarfile, joinpath(artifact_dir, tarname))
    end
    tarball_hash = archive_artifact(hash_, joinpath(tarfile))
    bind_artifact!(artifacts_toml, label, hash_,
                   download_info = [ (url, tarball_hash) ], lazy=true, force=true)
end



#---
# testing the artifacts

using ACE, Pkg, JuLIP
using Pkg.Artifacts

#---

acedata = joinpath(artifact"acedata_v0.8.0", "v0.8.0.tar.gz")
testsdata = joinpath(artifact"acedata_v0.8.0", "v0.8.0.tar.gz", "tests")

D = JuLIP.load_dict(joinpath(testsdata, "randship_v05.json"))

isfile(joinpath(testsdata, "randship_v05.json"))



const FONTSDIR = abspath(normpath(joinpath(artifact"fonts", "FIGletFonts-0.5.0", "fonts")))
