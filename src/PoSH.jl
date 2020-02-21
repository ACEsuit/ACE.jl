module PoSH


using Reexport
@reexport using JuLIP

include("aux.jl")
include("prototypes.jl")

# specification of the radial basis
include("pairpots/jacobi.jl")
include("pairpots/transforms.jl")

include("pairpots/basis.jl")
include("pairpots/calculator.jl")
include("pairpots/repulsion.jl")

include("pairpots/orthpolys.jl")
include("pairpots/one_orthogonal.jl")

# specification of the angular basis
include("sphericalharmonics.jl")
include("rotations.jl")

# basis specification: subsets of the full expansion
include("basisspecs.jl")

# implements the A functions ∏A functions
include("Alist.jl")

# SHIPBasis definition
include("basis.jl")

# SHIP interatomic potential definition
include("fast.jl")
include("real.jl")

include("regularisers.jl")

include("descriptors.jl")

include("utils.jl")

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
include("extras/convertc2r.jl")
include("extras/compressA.jl")


# ------ polynomials with cylindrical synmmetry (bonds)

include("bonds/bonds.jl") 

end # module
