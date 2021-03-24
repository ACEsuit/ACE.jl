



# prototypes for space transforms and cutoffs
function transform end
function transform_d end
function inv_transform end

function alloc_B end
function alloc_dB end
function fltype end
function alloc_temp end
function alloc_temp_d end
function evaluate! end
function evaluate_d! end
function write_dict end
function read_dict end

abstract type ACEBasis end
abstract type OneParticleBasis{T} <: ACEBasis end
abstract type ScalarACEBasis <: ACEBasis end

alloc_temp(::ACEBasis, args...) = nothing
alloc_temp_d(::ACEBasis, args...) = nothing

evaluate(basis::ACEBasis, args...) =
      evaluate!(alloc_B(basis), alloc_temp(basis), basis, args...)

evaluate_d(basis::ACEBasis, args...) =
      evaluate_d!(alloc_dB(basis), alloc_temp_d(basis), basis, args...)


# Some overloading to enable AD for some cases
#   TODO -> this needs to be extended..

evaluate(basis::ScalarACEBasis, args...) =
      evaluate!(alloc_B(basis, args...), alloc_temp(basis), basis, args...)

evaluate_d(basis::ScalarACEBasis, args...) =
      evaluate_d!(alloc_B(basis, args...), alloc_temp(basis), basis, args...)


# ------------------------------------------------------------
#  Abstract polynomial degree business

"""
`AbstractDegree` : object specifying a degree can be called via
`degree(D, arg)` or via `D(arg)`
"""
abstract type AbstractDegree end

(D::AbstractDegree)(args...) = degree(D, args...)

"""
`function degree(D::AbstractDegree, arg)` : compute some notion of degree of
the `arg` argument.
"""
function degree end

"""
interface functions for `OneParticleBasis`
"""
function add_into_A! end

"""
interface functions for `OneParticleBasis`
"""
function add_into_A_dA! end

"""
`function scaling(b, p)`:

a scaling factor for a basis functions ϕ, which gives a rought estimate on
   the magnitude of ∇ᵖϕ e.g.,
```
ϕ = r^n Ylm
```
has scaling factor `n^p + l^p`, though sharper estimates are also possible.
"""
function scaling end



using LinearAlgebra: Diagonal

diagonal_regulariser(basis; diff = 0) = Diagonal(scaling(basis, diff))

"""
every scalar basis must implement this
"""
function rand_radial end




struct NaiveTotalDegree end

# for a one-particle basis function
degree(b::NamedTuple, Deg::NaiveTotalDegree, basis::OneParticleBasis) =
      degree(b, basis)

# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
degree(bb, Deg::NaiveTotalDegree, basis::OneParticleBasis) =
      length(bb) == 0 ? 0 : sum( degree(b, basis) for b in bb )
