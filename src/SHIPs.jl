
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

module SHIPs

const IntS = Int32

using Reexport
@reexport using JuLIP

include("aux.jl")
include("prototypes.jl")

# specification of the radial basis
include("jacobi.jl")
include("transforms.jl")

# specification of the angular basis
include("sphericalharmonics.jl")

# body-order specific code: filter_tuple, _Bcoeff
include("bodyorders.jl")

# basis specification: subsets of the full expansion
include("basisspecs.jl")

# SHIPBasis definition
include("basis.jl")

# SHIP interatomic potential definition
include("fast.jl")

end # module
