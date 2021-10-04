
export SimpleSparseBasis, SparseBasis, MaxBasis 



# prototypes for space transforms and cutoffs
function transform end
function transform_d end
function transform_dd end
function inv_transform end


# ------------------------------------------------------------
#  Abstract polynomial degree business

"""
`AbstractBasisSelector` : object specifying how a finite basis is selected from 
the infinite space of symmetric polynomials.
"""
abstract type AbstractBasisSelector end

abstract type DownsetBasisSelector <: AbstractBasisSelector end 



"""
interface functions for `OneParticleBasis`
"""
function add_into_A! end

"""
interface functions for `OneParticleBasis`
"""
function add_into_A_dA! end


"""
`function degree` : compute some notion of degree of
the `arg` argument.
"""
function degree end

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

"""
No constraints on the basis - this selects that largest possible basis 
subject to additional constraints baked into the one-particle basis. 
In practise this should be equivalent to a naive max-norm basis selection.
"""
struct MaxBasis <: DownsetBasisSelector
end

isadmissible(b, bsel::MaxBasis, basis::OneParticleBasis) = true

"""
`struct SimpleSparseBasis`: 

The most basic form of a sparse basis selection, using the total degree. 
Only the maximum correlation order and maximum degree may be specified. 
This should only be used for testing. 
"""
struct SimpleSparseBasis <: DownsetBasisSelector
   maxorder::Int
   maxdeg::Float64
end


# for a one-particle basis function
degree(b::NamedTuple, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      degree(b, basis)

# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
degree(bb, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      length(bb) == 0 ? 0 : sum( degree(b, basis) for b in bb )

isadmissible(b::NamedTuple, Bsel::SimpleSparseBasis, basis::OneParticleBasis) = 
      (degree(b, Bsel, basis) <= Bsel.maxdeg)

isadmissible(bb, Bsel::SimpleSparseBasis, basis::OneParticleBasis) = 
      (degree(bb, Bsel, basis) <= Bsel.maxdeg) && length(bb) <= Bsel.maxorder

maxorder(Bsel::SimpleSparseBasis) = Bsel.maxorder 


"""
`SparseBasis`: probably the standard basis selector enabling weighted degree
functions and varying degree for different correlation orders. 
Need to add documentation; for now look at the code for `degree` to see how 
the weight and degree dictionaries affect the definition of degree.  
"""
struct SparseBasis <: DownsetBasisSelector
   maxorder::Int
   weight::Dict{Symbol, Float64}
   degree::Dict{Any, Float64}
   p::Float64
end

maxorder(Bsel::SparseBasis) = Bsel.maxorder 

isadmissible(b::NamedTuple, Bsel::SparseBasis, basis::OneParticleBasis) = 
      (degree(b, Bsel, basis) <= _maxdeg(Bsel, 0))

function isadmissible(bb, Bsel::SparseBasis, basis::OneParticleBasis) 
   ord = length(bb) 
   return (degree(bb, Bsel, basis) <= _maxdeg(Bsel, ord)) && ord <= Bsel.maxorder
end

_maxdeg(Bsel::SparseBasis, ord::Integer) = 
      haskey(Bsel.degree, ord) ? Bsel.degree[ord] : Bsel.degree["default"]

# for a one-particle basis function
degree(b::NamedTuple, Deg::SparseBasis, basis::OneParticleBasis) =
      degree(b, basis, Deg.weight)

# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
degree(bb, Bsel::SparseBasis, basis::OneParticleBasis)  =  (
      length(bb) == 0 ? 0.0 
                      : norm( degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p ) )


