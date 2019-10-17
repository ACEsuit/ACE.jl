module SHIPs

const IntS = Int32

import JuLIP.Potentials: z2i

using Reexport
@reexport using JuLIP

include("aux.jl")
include("prototypes.jl")

# specification of the radial basis
include("jacobi.jl")
include("transforms.jl")

# specification of the angular basis
include("sphericalharmonics.jl")
include("rotations.jl")

# basis specification: subsets of the full expansion
include("basisspecs.jl")

# SHIPBasis definition
include("basis.jl")

# SHIP interatomic potential definition
include("fast.jl")

# include("regularisers.jl")

# include("descriptors.jl")

include("utils.jl")


include("Alist.jl")

include("basis_new.jl")

end # module
