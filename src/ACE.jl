
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
import ACEbase: evaluate, evaluate_d, gradtype, valtype, 
                acquire_B!, release_B!, acquire_dB!, release_dB!, 
                ACEBasis, acquire!, release! 


# TODO: gradtype should have a standard fallback 


abstract type AbstractACEModel end 

abstract type AbstractProperty end

function coco_init end
function coco_zeros end
function coco_filter end
function coco_dot end
function coco_type end 

# TODO 
# * decide on rand(basis) interface

# * move these the following definitions to ACEbase
function _rrule_evaluate end 
function _rrule_evaluate_d end 

getlabel(basis::ACEBasis) = hasproperty(basis, :label) ? basis.label : ""


"""
This function is crucial to ACE internals. It implements the operation 
```
(x, y) -> ∑_i x[i] * y[i]
```
i.e. like `dot` but without taking conjugates. 
"""
contract(X1::AbstractVector, X2::AbstractVector) = 
            sum(contract(x1, x2) for (x1, x2) in zip(X1, X2))
            
contract(x1::Union{Number, AbstractProperty}, 
         x2::Union{Number, AbstractProperty}) = x1 * x2 

contract(X1::AbstractVector, x2::Union{Number, AbstractProperty}) = X1 * x2
contract(x1::Union{Number, AbstractProperty}, X2::AbstractVector) = x1 * X2

"""
sum of squares (without conjugation!)
"""
sumsq(x) = contract(x, x)

"""
norm-squared, i.e. sum xi * xi' 
"""
normsq(x) = dot(x, x)



include("auxiliary.jl")
include("transforms/lambdas.jl")

include("pools.jl")
include("ad.jl")
include("chain.jl")


include("rotations3d.jl")
using ACE.Rotations3D

include("polynomials/wigner.jl")

include("states.jl")
include("symmetrygroups.jl")
include("properties.jl")

contract(X1::AbstractVector{<: DState}, x2::DState) = contract.(X1, Ref(x2))


include("prototypes.jl")


# methods to transform stuff: 
#  - a state X into a variable that can be fed into a basis 
#  - distance transforms 
#  - transformations of an inner basis, e.g. linear transformations
include("transforms/transforms.jl"); @reexport using ACE.Transforms

# basic polynomial building blocks
include("polynomials/sphericalharmonics.jl")
include("polynomials/orthpolys.jl"); @reexport using ACE.OrthPolys

# 1p basis wrappers 
include("b1pcomponent.jl")
include("b1pcomponents/Rn.jl")
include("b1pcomponents/Ylm.jl")
include("b1pcomponents/scal.jl")
include("b1pcomponents/trig.jl")

# todo - move this once we get to it... 
include("discrete1pbasis.jl")

# the main product 1p basis 
include("product_1pbasis.jl")


# basis selectors used to specify finite subsets of basis functions
include("basisselectors.jl")
# ... amongst other things used to initialize sparse basis sets 
include("sparsegrids.jl")


# the permutation-invariant, and symmerized bases
include("pibasis.jl")
include("symmbasis.jl")


# some experimental stuff  
include("multiplier.jl")

# models and model evaluators

include("linearmodel.jl")

include("evaluator.jl")
# include("grapheval.jl")

include("random.jl")
@reexport using ACE.Random


include("utils.jl")
@reexport using ACE.Utils

include("testing/testing.jl")

# include("ad.jl")


# ---------------- some extra experimental dispatching

evaluate(basis::SymmetricBasis, Xs::AbstractVector) = 
      evaluate(basis, ACEConfig(Xs))

evaluate_d(basis::SymmetricBasis, Xs::AbstractVector) = 
      evaluate_d(basis, ACEConfig(Xs))

evaluate(model::LinearACEModel, Xs::AbstractVector) = 
      evaluate(model, ACEConfig(Xs))

grad_config(model::LinearACEModel, Xs::AbstractVector) = 
      grad_config(model, ACEConfig(Xs))

grad_params(model::LinearACEModel, Xs::AbstractVector) = 
      grad_params(model, ACEConfig(Xs))

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
