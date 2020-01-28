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

# include("pairpots/orthpolys.jl")

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
# include("real.jl")

include("regularisers.jl")

include("descriptors.jl")

include("utils.jl")

# ===== NEW STUFF 
# bond energies
# include("bonds/bonds.jl")

end # module
