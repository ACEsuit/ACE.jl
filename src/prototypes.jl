
# We use IntS for all index integer types for all SHIPs.jl basis types
# The `S` originates from the previous name of the package, `SHIPs.jl`
const IntS = Int32

import JuLIP.Potentials: z2i

import JuLIP: energy, forces, virial, alloc_temp, alloc_temp_d, cutoff,
              evaluate, evaluate_d,
              energy!, forces!, virial!, evaluate!, evaluate_d!

import JuLIP.MLIPs: IPBasis, alloc_B, alloc_dB

import Base: Dict, convert, ==

# prototypes for space transforms and cutoffs
function transform end
function transform_d end
function fcut end
function fcut_d end
