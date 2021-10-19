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
Special type of DownsetBasisSelector where basis functionts must be contained in the ball
   pnorm(bb) <= maxdeg
"""
abstract type PBallSelector <: DownsetBasisSelector end

degree(bb, Bsel::PBallSelector, basis::OneParticleBasis)  =  (
      length(bb) == 0 ? 0.0
                      : norm( degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p ) )


"""
`SparseBasis`: probably the standard basis selector enabling weighted degree
functions and varying degree for different correlation orders.
Need to add documentation; for now look at the code for `degree` to see how
the weight and degree dictionaries affect the definition of degree.
"""
struct SparseBasis <: PBallSelector
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



"""
`CategoryBasisSelector`: sam as `SparseBasis` but allows in addition to specify
maximum correlation orders for within-category correlations, i.e.,

   `maxorder_dict`
Simple basis selector enabling weighted degree
functions and varying degree for different correlation orders, and additional constraint
   to within-category
"""
struct CategoryBasisSelector <: PBallSelector
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

#

struct BasisSelectorIntersection <: DownsetBasisSelector
   Bselectors::Vector{DownsetBasisSelector}
end

Base.iterate(Intersection::BasisSelectorIntersection, args...) = Base.iterate(Intersection.Bselectors, args...)

function intersect(Intersection::BasisSelectorIntersection, Bsel2::DownsetBasisSelector)
   return BasisSelectorIntersection(append!(Intersection.Bselectors, Bsel2))
end

function intersect(Bsel1::DownsetBasisSelector, Bsel2::DownsetBasisSelector)
   return BasisSelectorIntersection([Bsel1, Bsel2])
end

function isadmissible(bb, Intersection::BasisSelectorIntersection, basis::OneParticleBasis)
   return all([isadmissible(bb, Bsel, basis) for Bsel in Intersection] )
end

maxorder(Intersection::BasisSelectorIntersection) = minimum([maxorder(Bsel) for Bsel in Intersection])

"""
`BasisSelectorCombine`: basis selector which allows to combine selection criteria of two basis selectors
*  `Bselectors`: vector of basis selectors whose selection criteria are combined
*  `degree`: dictionary that for each correlation order specifies the maximum degree of admissible configurations
*  `p`: value that specifies the p-norm according to which the total degree of basis functions is computed
If the constructor
```
BasisSelectorCombine(Bselectors::Vector{DownsetBasisSelector})
```
is used, then the resulting basis selector returns true for all basis functions that satisfy the
admissibility criteria of each selector in `Bselectors`. (The resulting type of basis selector does not implement the function
degree and such strictly speaking is not a subtype of `PBallSelector`)

If the constructor
```
BasisSelectorCombine(Bselectors::Vector{DownsetBasisSelector}, maxdegree::Int, p::Int)
```
is used, then the resulting basis selector returns true for all basis functions that satisfy the
admissibility criteria of each selector in `Bselectors` and whose totall degree is smaller than the correponding entry in degree
"""


struct BasisSelectorCombine <: PBallSelector
   Bselectors::Vector{DownsetBasisSelector}
   degree::Dict{Any, Float64}
   p::Int
end


BasisSelectorCombine(Bselectors::Vector{DownsetBasisSelector}) = BasisSelectorCombine(Bselectors, nothing, 1)
BasisSelectorCombine(Bselectors::Vector{DownsetBasisSelector}, maxdegree::Int, p::Int) = BasisSelectorCombine(Bselectors, Dict("default" => maxdegree), p)




Base.iterate(Intersection::BasisSelectorCombine, args...) = Base.iterate(BasisSelectorCombine, args...)

degree(b::NamedTuple, Intersection::BasisSelectorCombine, basis::OneParticleBasis) =
   prod([degree(b, Bsel, basis) for Bsel in Intersection])


function intersect(Intersection::BasisSelectorCombine, Bsel2::PBallSelector)
   return BasisSelectorCombine(append!(Intersection.Bselectors, Bsel2))
end

function intersect(Bsel1::PBallSelector, Bsel2::PBallSelector)
   return BasisSelectorCombine([Bsel1, Bsel2])
end

function isadmissible(bb, Intersection::BasisSelectorCombine, basis::OneParticleBasis)
   return all([isadmissible(bb, Bsel, basis) for Bsel in Intersection] ) && Intersection.degree != nothing &&
      degree(bb, Bsel, basis) <= _maxdeg(Bsel, length(bb))
end

maxorder(Intersection::BasisSelectorCombine) = minimum([maxorder(Bsel) for Bsel in Intersection])

_maxdeg(Bsel::BasisSelectorCombine, ord::Integer) =
         haskey(Bsel.degree, ord) ? Bsel.degree[ord] : Bsel.degree["default"]
