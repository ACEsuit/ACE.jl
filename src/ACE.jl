
module ACE

using Base: NamedTuple
using Reexport


# external imports that are useful for all submodules
include("imports.jl")

@extimports
@baseimports

# TODO 
# - move to imports

using ACEbase.ObjectPools: acquire!, release!, VectorPool
using ForwardDiff: derivative
import ChainRules: rrule, ZeroTangent, NoTangent
import ACEbase: evaluate, evaluate_d 
import ACEbase: gradtype, valtype
import ACEbase: acquire_B!, release_B!, acquire_dB!, release_dB! 


# TODO: gradtype should have a standard fallback 


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

# three specific 1p-bases that are always useful
include("Ylm1pbasis.jl")
include("Rn1pbasis.jl")
include("scal1pbasis.jl")

include("product_1pbasis.jl")

include("sparsegrids.jl")

# the permutation-invariant, and symmerized bases
include("symmetrygroups.jl")
include("pibasis.jl")
include("symmbasis.jl")


# models and model evaluators 

include("linearmodel.jl")

include("evaluator.jl")
# include("grapheval.jl")

include("random.jl")
@reexport using ACE.Random


include("utils.jl")
@reexport using ACE.Utils



include("testing/testing.jl")


include("ad.jl")


end # module


# TODO: 
# # orthogonal basis
# include("orth.jl")

# include("utils/importv5.jl")
# include("compat/compat.jl")
# include("export/export.jl")
# - bond model
# - pure basis
# - real basis
# - regularisers
# - descriptors
# - random potentials
# TODO -> move elsewhere!!!
# include("rpi/rpi_degrees.jl")
