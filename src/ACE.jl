
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module ACE

using Reexport
@reexport using JuLIP

# external imports that are useful for all submodules
include("extimports.jl")
include("julip_imports.jl")

include("states.jl")

include("auxiliary.jl")

include("prototypes.jl")


# basic polynomial building blocks
include("polynomials/sphericalharmonics.jl")
include("polynomials/transforms.jl"); @reexport using ACE.Transforms
include("polynomials/orthpolys.jl"); @reexport using ACE.OrthPolys

# The One-particle basis is the first proper building block
include("oneparticlebasis.jl")
include("species_1pbasis.jl")

# include("grapheval.jl")

# the permutation-invariant basis: this is a key building block
# for other bases but can also be a useful export itself
# include("pibasis.jl")

# include("pipot.jl")
#
# # rotation-invariant site potentials (incl the ACE model)
# include("rpi/rpi.jl")
# @reexport using ACE.RPI
#
# # orthogonal basis
# include("orth.jl")
#
# # pair potentials + repulsion
# include("pairpots/pair.jl");
# @reexport using ACE.PairPotentials
#
# # lots of stuff related to random samples:
# #  - random configurations
# #  - random potentials
# #  ...
# include("random.jl")
# @reexport using ACE.Random
#
#
# include("utils.jl")
# @reexport using ACE.Utils
#
# include("utils/importv5.jl")
#
#
# include("compat/compat.jl")
#
#
# include("export/export.jl")
#
#
# include("testing/testing.jl")

# - bond model
# - pure basis
# - real basis
# - regularisers
# - descriptors
# - random potentials


end # module
