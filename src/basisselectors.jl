

# Interface 

const Onepb = NamedTuple 
const Prodb = AbstractVector{<: NamedTuple}

"""
`AbstractBasisSelector` : object specifying how a finite basis is selected from
the infinite space of symmetric polynomials. This type is pretty superfluous 
for now since all basis selector and basis selection algorithms we admit so 
far require Downsets. 
"""
abstract type AbstractBasisSelector end

"""
`DownsetBasisSelector` : must implemented a positive valued `level` function 
that is monotonically increasing with respect to the lexiographic ordering of 
the basis functions.
"""
abstract type DownsetBasisSelector <: AbstractBasisSelector end

function maxorder

"""
* `level(b::Onepb, Bsel::DownsetBasisSelector, basis::OneParticleBasis)`
* `level(bb::Prodb, Bsel::DownsetBasisSelector, basis::OneParticleBasis)`

The first version specifies the level function for 
"""
function level end 

function maxlevel end 


"""
`level1(b::Onepb, Bsel::DownsetBasisSelector, basis::OneParticleBasis)`

A specialized version of `level` to construct the 1-particle basis (cf. 
`init1pspec!`). Fallback is to just call the `level` function, but this gives 
some additional flexibility to ensure that the conditions of the `level` 
framework are all justified, see docs for more detail. 
"""
level1(b::Onepb, Bsel::DownsetBasisSelector, basis::OneParticleBasis) = 
      level(b, Bsel, basis)


maxlevel1(Bsel::DownsetBasisSelector, basis::OneParticleBasis) = 
      maxlevel1(Bsel, basis)



"""
`filter(b_or_bb, Bsel::AbstractBasisSelector, basis::OneParticleBasis)`

After a down-set basis has been constructed, it can still be filtered, which 
allows us to construct basis sets that aren't downsets but not too far away from 
downsets. The main application is to enfore the constraints on m and l channels 
arising from the symmetries. 

Fallback implementation always returns `true`. 
"""
filter(b_or_bb, Bsel::AbstractBasisSelector, basis::OneParticleBasis) = true


filter(b, Bsel::DownsetBasisSelector, basis::OneParticleBasis) = true


"""
No constraints on the basis - this selects that largest possible basis
subject to additional constraints baked into the one-particle basis.
In practise this should be equivalent to a naive max-norm basis selection, 
and likely never used in practise. 
"""
struct MaxBasis <: DownsetBasisSelector
end

level(b::Onepb, Bsel::MaxBasis, basis::OneParticleBasis) = 
      degree(b, basis) 

level(bb::Prodb, Bsel::MaxBasis, basis::OneParticleBasis) = 
      length(bb) == 0 ? 0 : sum(b -> level(b, Bsel, basis), bb)

maxlevel(Bsel::MaxBasis, basis::OneParticleBasis) = Inf



"""
`struct SimpleSparseBasis`:

The most basic form of a sparse basis selection, using the total degree.
Only the maximum correlation order and maximum degree may be specified.
This should primarily be used for testing.
"""
struct SimpleSparseBasis <: DownsetBasisSelector
   maxorder::Int
   maxdeg::Float64
end

level(b::Onepb, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      degree(b, basis)

level(bb::Prodb, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      length(bb) == 0 ? 0 : sum( degree(b, basis) for b in bb )

maxlevel(Bsel::SimpleSparseBasis, args...) = 
      Bsel.maxdeg

maxorder(Bsel::SimpleSparseBasis, args...) = 
      Bsel.maxorder




"""
`AbstractSparseBasis`: Super-type for sparse basis selection as sub-levelsets of
the levelset function `level` and corresponding (possibly order-dependent)
levels provided in the dictionary `maxdegs::Dict{Any, Float64}`. In the default
implementation the levelset function and the degree function are identical.

Basis functions are selected in two steps. First, "admissible" basis 
specifications are generated as a sub-levelset of the leveset function using the 
implementation of the function `gensparse`.  After that, basis functions that do
not satisfy the conditons implemented in the function `filter` are removed from 
the basis set.
"""
abstract type AbstractSparseBasis <: DownsetBasisSelector end

maxorder(Bsel::AbstractSparseBasis) = Bsel.maxorder

# Default implementation of the degree function:
# for a one-particle basis function
degree(b::NamedTuple, Deg::AbstractSparseBasis, basis::OneParticleBasis) =
      degree(b, basis, Deg.weight)

# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
degree(bb, Bsel::AbstractSparseBasis, basis::OneParticleBasis)  =  (
      length(bb) == 0 ? 0.0
                      : norm( degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p ) )

# In the default implementation the levelset function and the degree function are identical 
level(b::NamedTuple, Bsel::AbstractSparseBasis, basis::OneParticleBasis) = 
      degree(b, Bsel, basis)

level(bb, Bsel::AbstractSparseBasis, basis::OneParticleBasis) =  (
      length(bb) == 0 ? 0.0
                      : norm( level.(bb, Ref(Bsel), Ref(basis)), Bsel.p ) )


# admissible specification are given as a sub-levelset of the levelset function 
isadmissible(b::NamedTuple, Bsel::AbstractSparseBasis, basis::OneParticleBasis) =
      (level(b, Bsel, basis) <= _maxdeg(Bsel, 1))

function isadmissible(bb, Bsel::AbstractSparseBasis, basis::OneParticleBasis)
   ord = length(bb)
   return ( level(bb, Bsel, basis) <= _maxdeg(Bsel, ord)) && ord <= Bsel.maxorder
end

_maxdeg(Bsel::AbstractSparseBasis, ord::Integer) =
      haskey(Bsel.maxdegs, ord) ? Bsel.maxdegs[ord] : Bsel.maxdegs["default"]




"""
`SparseBasis`: simplest/minimal implementation of an `AbstractSparseBasis`.

"""
struct SparseBasis <: AbstractSparseBasis
   maxorder::Int
   weight::Dict{Symbol, Float64}
   maxdegs::Dict{Any, Float64}
   p::Float64
end

function SparseBasis(maxorder::Int;  p = 1, 
                                          weight = Dict(:l => 1.0, :n => 1.0), 
                                          default_maxdeg = nothing, 
                                          maxdegs = nothing
                                          ) 
   if (default_maxdeg != nothing) && (maxdegs == nothing )
      SparseBasis(maxorder, weight, 
                  Dict{Any, Float64}("default" => default_maxdeg), 
                  p)
   elseif (default_maxdeg == nothing) && (maxdegs != nothing)
      SparseBasis(maxorder, weight, maxdegs, p)
   else
      @error """Either both or neither optional arguments `maxdegs` and 
                `default_maxdeg` were provided. To avoid ambiguity ensure that 
                exactly one of these arguments is provided."""
   end
end

"""
`CategorySparseBasis`: extension of `SparseBasis` that implements a 
constraint on the correlation orders for within-category correlations, i.e., for
each category `cat` contained in OneParticleBasis associated with the symbol 
`isym`, it is required that the within-category correlation order `corr[cat]`
satisfies
```julia 
minorder_dict[cat] <= corr[cat] <= maxorder_dict[cat].
```
"""
struct CategorySparseBasis <: AbstractSparseBasis
   maxorder::Int
   weight::Dict{Symbol, Float64}
   maxdegs::Dict{Any, Float64}
   p::Float64
   isym::Symbol
   minorder_dict::Dict{Any, Int}
   maxorder_dict::Dict{Any, Int}
   weight_cat::Dict{Any, Float64} 
end


function CategorySparseBasis(maxorder::Integer, isym::Symbol, categories;  
                             p = 1, 
                             weight = Dict{Symbol, Float64}(), 
                             default_maxdeg = nothing,
                             maxdegs = nothing,
                             minorder_dict = Dict{Any, Float64}(),
                             maxorder_dict = Dict{Any, Float64}(),
                             weight_cat = Dict(c => 1.0 for c in categories), 
                             ) 
   if (default_maxdeg != nothing) && (maxdegs == nothing )
      CategorySparseBasis(maxorder, weight, 
                          Dict{Any, Float64}("default" => default_maxdeg), 
                          p, isym, minorder_dict, maxorder_dict, weight_cat)
   elseif (default_maxdeg == nothing) && (maxdegs != nothing)
      CategorySparseBasis(maxorder, weight, maxdegs, p, isym, minorder_dict, 
                          maxorder_dict, weight_cat)
   else
      @error """Either both or neither optional arguments `maxdegs` and 
                `default_maxdeg` were provided. To avoid ambiguity ensure that 
                exactly one of these arguments is provided."""
   end
end

maxorder(Bsel::CategorySparseBasis, category) = Bsel.maxorder_dict[category]
minorder(Bsel::CategorySparseBasis, category) = Bsel.minorder_dict[category]

function filter(bb, Bsel::CategorySparseBasis, basis::OneParticleBasis) 
   ord = length(bb)
   # General order constrain that must be satisfied:
   ord_constr = ord <= Bsel.maxorder
   # Category-weighted degree constraint:
   level_set_constr = (cat_weighted_degree(bb, Bsel, basis) <= _maxdeg(Bsel, ord)) 

   # auxiliary function to count the number of 1pbasis functions in bb 
   # for which b.isym == s.
   num_b_is_(s) = sum([(getproperty(b, Bsel.isym) == s) for b in bb])

   # Within category min correlation order constaint:
   cond_ord_cats_min = all( num_b_is_(s) >= minorder(Bsel, s)
                            for s in keys(Bsel.minorder_dict) )
   # Within category max correlation order constaint:   
   cond_ord_cats_max = all( num_b_is_(s) <= maxorder(Bsel, s)
                            for s in keys(Bsel.minorder_dict) )

   return ord_constr && level_set_constr && cond_ord_cats_min && cond_ord_cats_max
end

# Category-weighted degree function
cat_weighted_degree(b::NamedTuple, Bsel::CategorySparseBasis, basis::OneParticleBasis) =
      degree(b, basis, Bsel.weight) * Bsel.weight_cat[getproperty(b, Bsel.isym)]

cat_weighted_degree(bb, Bsel::AbstractSparseBasis, basis::OneParticleBasis) = (
      length(bb) == 0 ? 0.0
                      : norm(cat_weighted_degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p)
      )
