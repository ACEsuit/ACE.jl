module PoSH

const IntS = Int32

import JuLIP.Potentials: z2i

import JuLIP: energy, forces, virial, alloc_temp, alloc_temp_d, cutoff

import JuLIP.Potentials:  evaluate, evaluate_d, evaluate!, evaluate_d!
import Base: Dict, convert, ==

function alloc_B end
function alloc_dB end


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

include("regularisers.jl")

include("descriptors.jl")

include("utils.jl")


end # module
