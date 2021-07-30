
module ACE

using Base: NamedTuple
using Reexport

include("objectpools.jl")
using ACE.ObjectPools: acquire!, release!

# TODO - could these have nice fall-backs? 
function acquire_B! end 
function release_B! end 
function acquire_dB! end 
function release_dB! end 



# external imports that are useful for all submodules
include("imports.jl")

@extimports
@baseimports

# TODO 
# - move to imports
# - retire alloc_B, alloc_dB 

using ForwardDiff: derivative
import ChainRules: rrule, ZeroTangent, NoTangent
import ACEbase: evaluate, evaluate_d 
import  ACEbase: gradtype, valtype, alloc_B, alloc_dB

# draft fallbacks 

acquire_B!(basis::ACEBasis, args...) = 
      acquire!(basis.B_pool, length(basis), valtype(basis, args...))
release_B!(basis::ACEBasis, B) = release!(basis.B_pool, B)

acquire_dB!(basis::ACEBasis, args...) = 
      acquire!(basis.dB_pool, length(basis), gradtype(basis, args...))
release_dB!(basis::ACEBasis, dB) = release!(basis.dB_pool, dB)


abstract type AbstractACEModel end 

abstract type AbstractProperty end

function coco_init end
function coco_zeros end
function coco_filter end
function coco_dot end

# TODO 
# * decide on rand(basis) interface
# * move these the following definitions to ACEbase
function _rrule_evaluate end 
function _rrule_evaluate_d end 

evaluate(basis::ACEBasis, args...) =  
      evaluate!( acquire_B!(basis, args...), basis, args... )

evaluate_d(basis::ACEBasis, args...) =  
      evaluate_d!( acquire_dB!(basis, args...), basis, args... )

evaluate_ed(basis::ACEBasis, args...) =  
      evaluate_ed!( acquire_B!(basis, args...), acquire_dB!(basis, args...), 
                    basis, args... )


include("auxiliary.jl")


include("rotations3d.jl")
using ACE.Rotations3D

include("polynomials/wigner.jl")

include("states.jl")
include("properties.jl")


include("prototypes.jl")


# basic polynomial building blocks
include("polynomials/sphericalharmonics.jl")
include("polynomials/transforms.jl"); @reexport using ACE.Transforms
include("polynomials/orthpolys.jl"); @reexport using ACE.OrthPolys

# The One-particle basis is the first proper building block
include("oneparticlebasis.jl")

# three specific 1p-bases that are useful
include("Ylm1pbasis.jl")
include("Rn1pbasis.jl")
include("scal1pbasis.jl")

include("product_1pbasis.jl")

include("sparsegrids.jl")


# the permutation-invariant basis: this is a key building block
# for other bases but can also be a useful export itself
include("pibasis.jl")

include("symmbasis.jl")

# models and model evaluators 


include("linearmodel.jl")

include("evaluator.jl")
# include("grapheval.jl")

# include("linearmodel.jl")


# # orthogonal basis
# include("orth.jl")

include("random.jl")
@reexport using ACE.Random


include("utils.jl")
@reexport using ACE.Utils

# include("utils/importv5.jl")
# include("compat/compat.jl")
# include("export/export.jl")


include("testing/testing.jl")


# - bond model
# - pure basis
# - real basis
# - regularisers
# - descriptors
# - random potentials
# TODO -> move elsewhere!!!
# include("rpi/rpi_degrees.jl")

include("ad.jl")

include("models/models.jl")

end # module
