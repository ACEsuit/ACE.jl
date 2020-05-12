
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module SHIPs


using Reexport
@reexport using JuLIP

using Parameters

include("aux.jl")
include("prototypes.jl")


# basic polynomial building blocks
include("polynomials/sphericalharmonics.jl")
include("polynomials/transforms.jl"); @reexport using SHIPs.Transforms
include("polynomials/orthpolys.jl"); @reexport using SHIPs.OrthPolys


# The One-particle basis is the first proper building block
include("oneparticlebasis.jl")

# the permutation-invariant basis: this is a key building block
# for other bases but can also be a useful export itself
include("pibasis.jl")
include("pipot.jl")

# rotation-invariant site potentials (incl the ACE model)
include("rpi/rpi.jl")

# pair potentials + repulsion
include("pairpots/pair.jl");
@reexport using SHIPs.PairPotentials



# - bond model
# - pure basis
# - real basis
# - regularisers
# - descriptors
# - random potentials

end # module
