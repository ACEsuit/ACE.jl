using ACE: OneParticleBasis
"""
`AbstractBasisSelector` : object specifying how a (possibly inifite) basis is selected from
the infinite space of symmetric polynomials.
"""
abstract type AbstractBasisSelector end

maxorder(Bsel::AbstractBasisSelector) = Inf

degree(bb, Bsel::AbstractBasisSelector, basis::OneParticleBasis) = 0

isadmissible(b, bsel::AbstractBasisSelector, basis::OneParticleBasis) = true

"""
No constraints on the basis - this selects that largest possible basis
subject to additional constraints baked into the one-particle basis.
In practise this should be equivalent to a naive max-norm basis selection.
"""
struct MaxBasis <: AbstractBasisSelector
end


"""
`struct SimpleSparseBasis`:

The most basic form of a sparse basis selection, using the total degree.
Only the maximum correlation order and maximum degree may be specified.
This should only be used for testing.
"""
struct SimpleSparseBasis <: AbstractBasisSelector
   maxorder::Int
   maxdeg::Float64
end


# for a one-particle basis function
degree(b::NamedTuple, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      degree(b, basis)

# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
degree(bb, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      length(bb) == 0 ? 0 : sum(degree(b, basis) for b in bb)

isadmissible(b::NamedTuple, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      (degree(b, Bsel, basis) <= Bsel.maxdeg)

isadmissible(bb, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      (degree(bb, Bsel, basis) <= Bsel.maxdeg) && length(bb) <= Bsel.maxorder

maxorder(Bsel::SimpleSparseBasis) = Bsel.maxorder

"""
Special type of PBallSelector where basis functionts must be contained in the ball
   pnorm(bb) <= maxdeg
"""
abstract type PBallSelector <: AbstractBasisSelector end

degree(bb, Bsel::PBallSelector, basis::OneParticleBasis)  =  (
length(bb) == 0 ? 0.0
                      : norm(degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p) )

degree(b::NamedTuple, Deg::PBallSelector, basis::OneParticleBasis) =
    degree(b, basis, Deg.weight)


isadmissible(b::NamedTuple, Bsel::PBallSelector, basis::OneParticleBasis) =
      (degree(b, Bsel, basis) <= _maxdeg(Bsel, 0))

function isadmissible(bb, Bsel::PBallSelector, basis::OneParticleBasis)
   ord = length(bb)
   return (degree(bb, Bsel, basis) <= _maxdeg(Bsel, ord)) 
end

_maxdeg(Bsel::PBallSelector, ord::Integer) =
      haskey(Bsel.degree, ord) ? Bsel.degree[ord] : Bsel.degree["default"]

"""
`SparseBasis`: probably the standard basis selector enabling weighted degree
functions and varying degree for different correlation orders.
Need to add documentation; for now look at the code for `degree` to see how
the weight and degree dictionaries affect the definition of degree.
"""
struct SparseBasis <: PBallSelector
   maxorder::Int
   weight::Dict{Symbol,Float64}
   degree::Dict{Any,Float64}
   p::Float64
end

maxorder(Bsel::SparseBasis) = Bsel.maxorder

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
   weight::Dict{Symbol,Float64}
   degree::Dict{Any,Float64}
   p::Float64
   isym::Symbol
   weight_cat::Dict{Any,Float64}
   maxorder_dict::Dict{Any,Int}
end

maxorder(Bsel::CategoryBasisSelector) = Bsel.maxorder
maxorder(Bsel::CategoryBasisSelector, category) = Bsel.maxorder_dict[category]

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


struct CategoryWeightedBasisSelector <: PBallSelector
   weight::Dict{Symbol,Float64}
   degree::Dict{Any,Float64}
   p::Float64
   isym::Symbol
   weight_cat::Dict{Any,Float64}
end
# for a one-particle basis function
degree(b::NamedTuple, Bsel::CategoryWeightedBasisSelector, basis::OneParticleBasis) =
      degree(b, basis, Bsel.weight) * Bsel.weight_cat[getproperty(b, Bsel.isym)]


struct CategoryConstraint <: AbstractBasisSelector
   isym::Symbol
   maxorder_dict::Dict{Any,Int}
end

maxorder(Bsel::CategoryConstraint, category) = Bsel.maxorder_dict[category]

isadmissible(b::NamedTuple, Bsel::CategoryConstraint, basis::OneParticleBasis) = true

function isadmissible(bb, Bsel::CategoryConstraint, basis::OneParticleBasis)
   cond_ord_cats = [
         sum([ getproperty(b, Bsel.isym) == s for b in bb ]) <= maxorder(Bsel, s)
                                 for s in keys(Bsel.maxorder_dict) ]

   return all(cond_ord_cats)
      
end

struct OrderConstraint <: AbstractBasisSelector
   maxorder::Int
end

maxorder(Bsel::OrderConstraint) = Bsel.maxorder

function isadmissible(bb, Bsel::OrderConstraint, basis::OneParticleBasis)
   return length(bb) <= maxorder(Bsel)
end

"""
`BasisSelectorCombine`: basis selector which allows to combine selection criteria of two basis selectors
*  `Bselectors`: vector of basis selectors whose selection criteria are combined
*  `degree`: dictionary that for each correlation order specifies the maximum degree of admissible configurations
*  `p`: value that specifies the p-norm according to which the total degree of basis functions is computed
If the constructor
```
BasisSelectorCombine(Bselectors::Vector{AbstractBasisSelector})
```
is used, then the resulting basis selector returns true for all basis functions that satisfy the
admissibility criteria of each selector in `Bselectors`. (The resulting type of basis selector may not implement the function
degree and such strictly speaking is not a subtype of `AbstractBasisSelector`)

If the constructor
```
BasisSelectorCombine(Bselectors::Vector{AbstractBasisSelector}, maxdegree::Int, p::Int)
```
is used, then the resulting basis selector returns true for all basis functions that satisfy the
admissibility criteria of each selector in `Bselectors` and whose totall degree is smaller than the correponding entry in degree
"""


struct Intersection <: AbstractBasisSelector
   Bselectors::Vector{AbstractBasisSelector}
end


#Intersection(Bselectors::Vector{AbstractBasisSelector}) = Intersection(Bselectors)

#Intersection(Bselectors::Vector{AbstractBasisSelector}, maxdegree::Int, p::Int) = Intersection(Bselectors, Dict("default" => maxdegree), p)


intersect(Bsel1::AbstractBasisSelector, Bsel2::AbstractBasisSelector) = Intersection([Bsel1, Bsel2])
intersect(Bsel1::Intersection, Bsel2::AbstractBasisSelector) = Intersection(push!(Bsel1.Bselectors, Bsel2))
intersect(Bsel1::Intersection, Bsel2::Intersection) = Intersection(cat(Bsel1.Bselectors,Bsel2.Bselectors))


#∩(Bsel1::AbstractBasisSelector, Bsel2::AbstractBasisSelector) = intersect(Bsel1, Bsel2)

Base.iterate(Bsel::Intersection, args...) = Base.iterate(Bsel.Bselectors, args...)
Base.length(Bsel::Intersection) = 1 #Base.length(Bsel.Bselectors)

degree(b::NamedTuple, Intersection::Intersection, basis::OneParticleBasis) =
   sum([degree(b, Bsel, basis) for Bsel in Intersection])


isadmissible(b::NamedTuple, Bsel::Intersection, basis::OneParticleBasis) = all([isadmissible(b, bsel, basis) for bsel in Bsel]) 
function isadmissible(bb, Bsel::Intersection, basis::OneParticleBasis)
   return all([isadmissible(bb, bsel, basis) for bsel in Bsel]) 
   #&& degree(bb, Bsel, basis) <= _maxdeg(Bsel, length(bb))
end

maxorder(Bsel::Intersection) = Int(minimum([maxorder(bsel) for bsel in Bsel.Bselectors]))

_maxdeg(Bsel::Intersection, ord::Integer) =
         haskey(Bsel.degree, ord) ? Bsel.degree[ord] : Bsel.degree["default"]
