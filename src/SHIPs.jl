
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


# this includes some utilities to specify different notion of degree
include("degrees.jl")

# The One-particle basis is the first proper building block
include("oneparticlebasis.jl")

# the permutation-invariant basis: this is a key building block
# for other bases but can also be a useful export itself
# include("pibasis.jl")



# the permutation-invariant basis is the second main building block
# and at the same time already a useful export
# include("PIBasis.jl")


# include("harmonics/rotations.jl")
# specification of the radial basis
# include("pairpots/basis.jl")
# include("pairpots/calculator.jl")
# include("pairpots/repulsion.jl")
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
