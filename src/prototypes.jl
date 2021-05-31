



# prototypes for space transforms and cutoffs
function transform end
function transform_d end
function inv_transform end


# ------------------------------------------------------------
#  Abstract polynomial degree business

"""
`AbstractDegree` : object specifying a degree can be called via
`degree(D, arg)` or via `D(arg)`
"""
abstract type AbstractDegree end

(D::AbstractDegree)(args...) = degree(D, args...)

"""
interface functions for `OneParticleBasis`
"""
function add_into_A! end

"""
interface functions for `OneParticleBasis`
"""
function add_into_A_dA! end


"""
`function degree(D::AbstractDegree, arg)` : compute some notion of degree of
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
`struct NaiveTotalDegree  end`

This should only be used for testing.
"""
struct NaiveTotalDegree  
end

# for a one-particle basis function
degree(b::NamedTuple, Deg::NaiveTotalDegree, basis::OneParticleBasis) =
      degree(b, basis)

# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
degree(bb, Deg::NaiveTotalDegree, basis::OneParticleBasis) =
      length(bb) == 0 ? 0 : sum( degree(b, basis) for b in bb )



struct TotalDegree
   weight::Dict{Symbol, Float64}
   degree::Dict{Any, Float64}
end

# for a one-particle basis function
degree(b::NamedTuple, Deg::TotalDegree, basis::OneParticleBasis) =
      degree(b, basis, Deg.weight)

# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
function degree(bb, Deg::TotalDegree, basis::OneParticleBasis) 
   if length(bb) == 0; return 0; end 
   len = length(bb)
   if haskey(Deg.degree, len)
      degfact = Deg.degree[len]
   elseif haskey(Deg.degree, "default")
      degfact = Deg.degree["default"]
   else 
      degfact = 1.0
   end
   return sum( degree(b, Deg, basis)  for b in bb ) * degfact
end



