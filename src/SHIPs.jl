
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module SHIPs


using Reexport
@reexport using JuLIP

include("aux.jl")

include("prototypes.jl")

# spherical harmonics, and codes related to 3d rotations
include("harmonics/sphericalharmonics.jl")
# include("harmonics/rotations.jl")


include("oneparticlebasis.jl")

# # specification of the radial basis
# include("pairpots/jacobi.jl")
# include("pairpots/transforms.jl")
# include("pairpots/basis.jl")
# include("pairpots/calculator.jl")
# include("pairpots/repulsion.jl")
# include("pairpots/orthpolys.jl")
#
#
# # basis specification: subsets of the full expansion
# include("basisspecs.jl")
#
# # implements the A functions ∏A functions
# include("Alist.jl")
#
# # SHIPBasis definition
# include("basis.jl")
# include("purebasis.jl")
#
# # SHIP interatomic potential definition
# include("fast.jl")
# include("real.jl")
#
# include("regularisers.jl")
#
# include("descriptors.jl")
#
# include("utils.jl")

# ===== NEW STUFF
# bond energies
# include("bonds/bonds.jl")

# OPTIONAL MODULES
# using Requires
#
# function _init_()
#    @require SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6" begin
#       include("convertc2r.jl")
#    end
# end
#  TODO: make conertc2r load only conditionally. The above seems to be
#        incorrect, probably used Requires incorrectly?
# include("extras/convertc2r.jl")
# include("extras/compressA.jl")


# # ------ polynomials with cylindrical synmmetry (bonds)
# include("bonds/bonds.jl")


# # ------ pure permutation invariance
# include("PIBasis.jl")
#
#
# # ---------------------- experimental
# include("experimental.jl")

end # module
