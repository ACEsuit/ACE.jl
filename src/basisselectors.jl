
"""
`AbstractBasisSelector` : object specifying how a finite basis is selected from
the infinite space of symmetric polynomials.
"""
abstract type AbstractBasisSelector end

"""
`DownsetBasisSelector` : must implemented a positive valued degree 
function `degree` that is monotonically increasing with respect to 
the lexiographic ordering of the basis functions.  
"""
abstract type DownsetBasisSelector <: AbstractBasisSelector end


filter(b, Bsel::DownsetBasisSelector, basis::OneParticleBasis) = true


"""
No constraints on the basis - this selects that largest possible basis
subject to additional constraints baked into the one-particle basis.
In practise this should be equivalent to a naive max-norm basis selection.
"""
struct MaxBasis <: DownsetBasisSelector
end

isadmissible(b::NamedTuple, Bsel::MaxBasis, basis::OneParticleBasis) = true
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
`SparseBasis`: Super-type for sparse basis selection as sub-levelsets of the levelset function `level` and corresponding (possibly order-dependent) 
levels provided in the dictionary `max_degrees::Dict{Any, Float64}`. In the default implementation the levelset function and the degree function are identical.

Basis functions are selected in two steps. First, "admissible" basis specifications are generated as a sub-levelset of the leveset function using the implementation of the function `gensparse`. 
After that, basis functions that do not satisfy the conditons implemented in the function `filter` are removed from the basis set.
"""

abstract type SparseBasis <: DownsetBasisSelector end

maxorder(Bsel::SparseBasis) = Bsel.maxorder

# Default implementation of the degree function:
# for a one-particle basis function
degree(b::NamedTuple, Deg::SparseBasis, basis::OneParticleBasis) =
      degree(b, basis, Deg.weight)

# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
degree(bb, Bsel::SparseBasis, basis::OneParticleBasis)  =  (
      length(bb) == 0 ? 0.0
                      : norm( degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p ) )

# In the default implementation the levelset function and the degree function are identical 
level(b::NamedTuple, Bsel::SparseBasis, basis::OneParticleBasis)  = degree(b, Bsel, basis) 
level(bb, Bsel::SparseBasis, basis::OneParticleBasis) =  (length(bb) == 0 ? 0.0
                      : norm( level.(bb, Ref(Bsel), Ref(basis)), Bsel.p ) )


# admissible specification are given as a sub-levelset of the levelset function 
isadmissible(b::NamedTuple, Bsel::SparseBasis, basis::OneParticleBasis) =
      (level(b, Bsel, basis) <= _maxdeg(Bsel, 0))

function isadmissible(bb, Bsel::SparseBasis, basis::OneParticleBasis)
   ord = length(bb)
   return ( level(bb, Bsel, basis) <= _maxdeg(Bsel, ord)) && ord <= Bsel.maxorder
end

_maxdeg(Bsel::SparseBasis, ord::Integer) =
      haskey(Bsel.maxdegs, ord) ? Bsel.maxdegs[ord] : Bsel.maxdegs["default"]




"""
`PNormSparseBasis`: simplest implementation of a `SparseBasis`.
"""
struct PNormSparseBasis <: SparseBasis
   maxorder::Int
   weight::Dict{Symbol, Float64}
   maxdegs::Dict{Any, Float64}
   p::Float64
end

function PNormSparseBasis(maxorder::Int;  p = 1, 
                                          weight = Dict(:l => 1.0, :n => 1.0), 
                                          default_maxdeg = nothing, 
                                          maxdegs = Dict{Any, Float64}()
                                          ) 
   if default_maxdeg !== nothing  maxdegs == Dict{Any, Float64}()
      PNormSparseBasis(maxorder, weight, Dict{Any, Float64}("default" => default_maxdeg), p)
   elseif default_maxdeg === nothing && maxdegs != Dict{Any, Float64}()
      PNormSparseBasis(maxorder, weight, maxdegs, p)
   else
      @error "Either both or neither optional arguments `maxdegs` and `default_maxdeg` were provided. To avoid ambiguity ensure that exactly one of these arguments is provided."
   end
end

"""
`CategorySparseBasis`: extension of `PNormSparseBasis` that implements a constraint on 
the correlation orders for within-category correlations, i.e., for each category `cat` contained in OneParticleBasis associated with 
   the symbol `isym`, it is required that the within-category correlation order `corr[cat]` satisfies
   ```
   minorder_dict[cat] <= corr[cat] <= maxorder_dict[cat].
   ```
"""
struct CategorySparseBasis <: SparseBasis
   maxorder::Int
   weight::Dict{Symbol, Float64}
   maxdegs::Dict{Any, Float64}
   p::Float64
   isym::Symbol
   minorder_dict::Dict{Any, Int}
   maxorder_dict::Dict{Any, Int}
   weight_cat::Dict{Any, Float64} 
end


function CategorySparseBasis(maxorder::Int,isym::Symbol,categories::AbstractArray;  p = 1, 
                                                                                    weight = Dict{Symbol, Float64}(), 
                                                                                    default_maxdeg = 10,
                                                                                    maxdegs = Dict{Any, Float64}(),
                                                                                    minorder_dict = Dict{Any, Float64}(),
                                                                                    maxorder_dict = Dict{Any, Float64}(),
                                                                                    weight_cat = Dict(c => 1.0 for c in categories), 
                                                                                    ) 
   if default_maxdeg !== nothing  maxdegs == Dict{Any, Float64}() 
      CategorySparseBasis(maxorder, weight, Dict{Any, Float64}("default" => default_maxdeg), p, isym, minorder_dict, maxorder_dict, weight_cat)
   elseif default_maxdeg === nothing && maxdegs != Dict{Any, Float64}()
      CategorySparseBasis(maxorder, weight, maxdegs, p, isym, minorder_dict, maxorder_dict, weight_cat)
   else
      @error "Either both or neither optional arguments `maxdegs` and `default_maxdeg` were provided. To avoid ambiguity ensure that exactly one of these arguments is provided."
   end
end

maxorder(Bsel::CategorySparseBasis, category) = Bsel.maxorder_dict[category]
minorder(Bsel::CategorySparseBasis, category) = Bsel.minorder_dict[category]

function filter(bb, Bsel::ConstrainedSparseBasis, basis::OneParticleBasis) 
   ord = length(bb)
   # General order constrain that must be satisfied:
   ord_constr = ord <= Bsel.maxorder
   # Category-weighted degree constraint:
   level_set_constr = (cat_weighted_degree(bb, Bsel, basis) <= _maxdeg(Bsel, ord)) 
   # Within category min correlation order constaint:
   cond_ord_cats_min = all([
      sum([ getproperty(b, Bsel.isym) == s for b in bb ]) >= minorder(Bsel, s)
                              for s in keys(Bsel.minorder_dict) ]  )
   # Within category max correlation order constaint:   
   cond_ord_cats_max = all([
      sum([ getproperty(b, Bsel.isym) == s for b in bb ]) <= maxorder(Bsel, s)
                              for s in keys(Bsel.maxorder_dict) ]  )                     

   return ord_constr && level_set_constr && cond_ord_cats_min && cond_ord_cats_max
end

# Category-weighted degree function
cat_weighted_degree(b::NamedTuple, Bsel::CategorySparseBasis, basis::OneParticleBasis) =
      degree(b, basis, Bsel.weight) * Bsel.weight_cat[getproperty(b, Bsel.isym)]

function cat_weighted_degree(bb, Bsel::SparseBasis, basis::OneParticleBasis)
   return (length(bb) == 0 ? 0.0
                        : norm( cat_weighted_degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p ) )
end
      



