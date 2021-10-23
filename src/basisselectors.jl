
"""
`AbstractBasisSelector` : object specifying how a finite basis is selected from
the infinite space of symmetric polynomials.
"""
abstract type AbstractBasisSelector end

abstract type DownsetBasisSelector <: AbstractBasisSelector end


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





"""
`CategoryBasisSelector`: sam as `SparseBasis` but allows in addition to specify
maximum correlation orders for within-category correlations, i.e.,

   `maxorder_dict`
Simple basis selector enabling weighted degree
functions and varying degree for different correlation orders, and additional constraint
   to within-category
Need to add documentation; for now look at the code for `degree` to see how
the weight and degree dictionaries affect the definition of degree.
"""
struct CategoryBasisSelector <: DownsetBasisSelector
   maxorder::Int
   maxorder_dict::Dict{Any, Int}
   isym::Symbol
   weight::Dict{Symbol, Float64}
   weight_cat::Dict{Any, Float64}
   degree::Dict{Any, Float64}
   p::Float64
end



maxorder(Bsel::CategoryBasisSelector) = Bsel.maxorder

maxorder(Bsel::CategoryBasisSelector, category) = Bsel.maxorder_dict[category]

_maxdeg(Bsel::CategoryBasisSelector, ord::Integer) =
         haskey(Bsel.degree, ord) ? Bsel.degree[ord] : Bsel.degree["default"]


isadmissible(b::NamedTuple, Bsel::CategoryBasisSelector, basis::OneParticleBasis) =
      (degree(b, Bsel, basis) <= _maxdeg(Bsel, 0))

function isadmissible(bb, Bsel::CategoryBasisSelector, basis::OneParticleBasis)
   cond_ord = length(bb) <= maxorder(Bsel)
   cond_ord_cats = [
         sum([ getproperty(b, Bsel.isym) == s for b in bb ]) <= maxorder(Bsel, s)
                                 for s in keys(Bsel.maxorder_dict) ]

   return (  (degree(bb, Bsel, basis) <= _maxdeg(Bsel, length(bb)))
             && cond_ord
             && all(cond_ord_cats)
         )
end


# for a one-particle basis function
degree(b::NamedTuple, Bsel::CategoryBasisSelector, basis::OneParticleBasis) =
      degree(b, basis, Bsel.weight) * Bsel.weight_cat[getproperty(b, Bsel.isym)]


# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
function degree(bb, Bsel::CategoryBasisSelector, basis::OneParticleBasis)
   if length(bb) == 0
      return 0.0
   else
      return norm( degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p )
   end
end
