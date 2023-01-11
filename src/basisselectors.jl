

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
`DownsetBasisSelector` : must implemented a non-negative valued `level` function 
that is monotonically increasing with respect to the lexiographic ordering of 
the basis functions.
"""
abstract type DownsetBasisSelector <: AbstractBasisSelector end

"""
return maximum correlation order of the basis selector. Note that the filter 
function could reduce that by a bit.
"""
function maxorder end 

"""
* `level(b::Onepb, Bsel::DownsetBasisSelector, basis::OneParticleBasis)`
* `level(bb::Prodb, Bsel::DownsetBasisSelector, basis::OneParticleBasis)`

The first version specifies the level function for 
"""
function level end 

"""
`maxlevel(Bsel, basis1p)`
"""
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
      maxlevel(Bsel, basis)



"""
`filter(b_or_bb, Bsel::AbstractBasisSelector, basis::OneParticleBasis)`

After a down-set basis has been constructed, it can still be filtered, which 
allows us to construct basis sets that aren't downsets but not too far away from 
downsets. The main application is to enfore the constraints on m and l channels 
arising from the symmetries. 

Fallback implementation always returns `true`. 
"""
filter(b_or_bb, Bsel::AbstractBasisSelector, basis::OneParticleBasis) = true


"""
No constraints on the basis - this selects that largest possible basis
subject to additional constraints baked into the one-particle basis.
In practise this should be equivalent to a naive max-norm basis selection, 
and likely never used in practise. 
"""
struct MaxBasis <: DownsetBasisSelector
   maxorder::Int 
end

maxorder(Bsel::MaxBasis) = Bsel.maxorder 

level(b::Onepb, Bsel::MaxBasis, basis::OneParticleBasis) = 
      degree(b, basis) 

level(bb::Prodb, Bsel::MaxBasis, basis::OneParticleBasis) = 
      length(bb) == 0 ? 0 : sum(b -> level(b, Bsel, basis), bb)

maxlevel(Bsel::MaxBasis, basis::OneParticleBasis) = Inf



"""
`struct SimpleSparseBasis`:

The most basic form of a sparse basis selection, using the total degree.
Only the maximum correlation order and maximum degree may be specified.
This should primarily be used for testing. Construct it via 
```julia
SimpleSparseBasis(maxorder, maxdegree)
```
Note that `maxdegree` is a basic total degree. 
"""
struct SimpleSparseBasis <: DownsetBasisSelector
   maxorder::Int
   maxlevel::Float64
end

level(b::Onepb, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      degree(b, basis)

level(bb::Prodb, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      length(bb) == 0 ? 0 : sum( degree(b, basis) for b in bb )

maxlevel(bb, Bsel::SimpleSparseBasis, args...) = 
      Bsel.maxlevel

maxlevel(Bsel::SimpleSparseBasis, args...) = 
      Bsel.maxlevel

maxorder(Bsel::SimpleSparseBasis, args...) = 
      Bsel.maxorder



"""
`AbstractSparseBasis`: Super-type for sparse basis selection as sub-levelsets of
the levelset function `level` and corresponding (possibly order-dependent)
levels provided in the dictionary `maxlevels::Dict{Any, Float64}`. In the default
implementation the levelset function and the degree function are identical.

Basis functions are selected in two steps. First, "admissible" basis 
specifications are generated as a sub-levelset of the leveset function using the 
implementation of the function `gensparse`.  After that, basis functions that do
not satisfy the conditons implemented in the function `filter` are removed from 
the basis set.
"""
abstract type AbstractSparseBasis <: DownsetBasisSelector end

maxorder(Bsel::AbstractSparseBasis) = Bsel.maxorder

level(b::Onepb, Bsel::AbstractSparseBasis, basis::OneParticleBasis) =
      degree(b, basis, Bsel.weight)

level(bb::Prodb, Bsel::AbstractSparseBasis, basis::OneParticleBasis)  =  (
      length(bb) == 0 ? 0.0
                      : norm( level.(bb, Ref(Bsel), Ref(basis)), Bsel.p ) )

maxlevel(bb::Prodb, Bsel::AbstractSparseBasis, basis::OneParticleBasis) = 
      maxlevel(length(bb), Bsel, basis)

maxlevel(ord::Integer, Bsel::AbstractSparseBasis, basis::OneParticleBasis) = 
      ( haskey(Bsel.maxlevels, ord) ? Bsel.maxlevels[ord] 
                                    : Bsel.maxlevels["default"] )

maxlevel(Bsel::AbstractSparseBasis, basis::OneParticleBasis) = 
      maximum( maxlevel(ord, Bsel, basis) for ord = 1:maxorder(Bsel) )


"""
`SparseBasis`: basic implementation of an `AbstractSparseBasis`.
- `maxorder` : maximum correlation order 
- `p` : degree (level) is computed via a weighted p-norm of the 1p basis 
      function degrees
- `maxlevels` : for each order `ord`, `maxlevels[ord]` is the maximum level  
      for basis functions of that order. if `maxlevels[ord]` is not 
      specified, then `maxlevels["default"]` is the fallback.
- `weight` : specify weights for the different channels of the 1p basis 
      functions, e.g., for an `Rn * Ylm` basis, one might specify 
      `weight[:n] = 1` and `weight[:l] = 2` to have more radial and 
      fewer angular basis functions. 
"""
struct SparseBasis <: AbstractSparseBasis
   maxorder::Int
   weight::Dict{Symbol, Float64}
   maxlevels::Dict{Any, Float64}
   p::Float64
end

function SparseBasis(;  maxorder::Integer = nothing, 
                        p = 1, 
                        weight = Dict(:l => 1.0, :n => 1.0), 
                        default_maxdeg = nothing, 
                        maxlevels = nothing ) 
   if (default_maxdeg != nothing) && (maxlevels == nothing )
      return SparseBasis(maxorder, weight, 
                         Dict{Any, Float64}("default" => default_maxdeg), 
                         p)
   elseif (default_maxdeg == nothing) && (maxlevels != nothing)
      SparseBasis(maxorder, weight, maxlevels, p)
   else
      error("""Either both or neither optional arguments `maxlevels` and 
               `default_maxdeg` were provided. To avoid ambiguity ensure that 
               exactly one of these arguments is provided.""")
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
   maxlevels::Dict{Any, Float64}
   p::Float64
   isym::Symbol
   minorder_dict::Dict{Any, Int}
   maxorder_dict::Dict{Any, Int}
   weight_cat::Dict{Any, Float64} 
end


function CategorySparseBasis(isym::Symbol, categories;  
                             maxorder::Integer = nothing, 
                             p = 1, 
                             weight = Dict{Symbol, Float64}(), 
                             default_maxdeg = nothing,
                             maxlevels::Dict{Any, Float64} = nothing,
                             minorder_dict = Dict{Any, Float64}(),
                             maxorder_dict = Dict{Any, Float64}(),
                             weight_cat = Dict(c => 1.0 for c in categories), 
                             ) 
   if (default_maxdeg != nothing) && (maxlevels == nothing )
      CategorySparseBasis(maxorder, weight, 
                          Dict{Any, Float64}("default" => default_maxdeg), 
                          p, isym, minorder_dict, maxorder_dict, weight_cat)
   elseif (default_maxdeg == nothing) && (maxlevels != nothing)
      CategorySparseBasis(maxorder, weight, maxlevels, p, isym, minorder_dict, 
                          maxorder_dict, weight_cat)
   else
      @error """Either both or neither optional arguments `maxlevels` and 
                `default_maxdeg` were provided. To avoid ambiguity ensure that 
                exactly one of these arguments is provided."""
   end
end

maxorder(Bsel::CategorySparseBasis, category) = Bsel.maxorder_dict[category]

minorder(Bsel::CategorySparseBasis, category) = Bsel.minorder_dict[category]

filter(b::ACE.Onepb, Bsel::CategorySparseBasis, basis::OneParticleBasis) = true

function filter(bb, Bsel::CategorySparseBasis, basis::OneParticleBasis) 
   # auxiliary function to count the number of 1pbasis functions in bb 
   # for which b.isym == s.
   num_b_is_(s) = sum([(getproperty(b, Bsel.isym) == s) for b in bb])

   # Within category min correlation order constaint:
   cond_ord_cats_min = all( num_b_is_(s) >= minorder(Bsel, s)
                            for s in keys(Bsel.minorder_dict) )
   # Within category max correlation order constaint:   
   cond_ord_cats_max = all( num_b_is_(s) <= maxorder(Bsel, s)
                            for s in keys(Bsel.maxorder_dict) )

   return cond_ord_cats_min && cond_ord_cats_max
end

# maxorder and maxlevel are inherited from the abstract interface 

level(b::Onepb, Bsel::CategorySparseBasis, basis::OneParticleBasis) =
      cat_weighted_degree(b, Bsel, basis)

level(bb::Prodb, Bsel::CategorySparseBasis, basis::OneParticleBasis)  =  
      cat_weighted_degree(bb, Bsel, basis)

# Category-weighted degree function
cat_weighted_degree(b::Onepb, Bsel::CategorySparseBasis, basis::OneParticleBasis) =
      degree(b, basis, Bsel.weight) * Bsel.weight_cat[getproperty(b, Bsel.isym)]

cat_weighted_degree(bb::Prodb, Bsel::CategorySparseBasis, basis::OneParticleBasis) = (
      length(bb) == 0 ? 0.0
                      : norm(cat_weighted_degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p)
      )


# --------------------------- 
# Some useful filters 

struct NoConstant 
end

(::NoConstant)(bb) = (length(bb) > 0)


"""
`EvenL`: selects all basis functions where the sum `L = sum_i l_i` of the degrees `l_i` of the spherical harmonics is even.   
"""
struct EvenL
      isym::Symbol
      categories
end
  
function (f::ACE.EvenL)(bb) 
      if isempty(bb)
            return true
      else
            suml(s) = sum( [getl(O3(), b) for b in bb if getproperty(b, f.isym) == s])
            return all(iseven(suml(s)) for s in f.categories)
      end
end


#=
"""
`DownsetIntersection`: Basis selector whose set of admissible specifications is the intersection 
of the sets of admissible specifications of the sparse basis selectors contained in the lists `DBsels` and `ABsels`.
"""
struct DownsetIntersection <: DownsetBasisSelector
   DBsels::Vector{DownsetBasisSelector}
   ABsels::Vector{AbstractBasisSelector}
   maxorder::Int
end

maxlevel(Bsel::DownsetIntersection, basis::OneParticleBasis) = 1.0 
maxorder(Bsel::DownsetIntersection) = Bsel.maxorder


function Base.intersect(Bsel1::DownsetIntersection,Bsel2::DownsetBasisSelector)
      return DownsetIntersection(vcat(Bsel1.DBsels,[Bsel2]), BSel1.ABsels, minimum([maxorder(Bsel1),maxorder(Bsel2)]))
end

function Base.intersect(Bsel1::DownsetIntersection,Bsel2::AbstractBasisSelector)
      return DownsetIntersection(Bsel1.DBsels, vcat(Bsel1.ABsels,[Bsel2]), minimum([maxorder(Bsel1),maxorder(Bsel2)]))
end

function Base.intersect(Bsel1::DownsetBasisSelector, Bsel2::AbstractBasisSelector) 
      Bsel = DownsetIntersection(Vector{DownsetBasisSelector}([]), Vector{AbstractBasisSelector}([]), Inf)
      return intersect(intersect(Bsel, Bsel1), Bsel2)
end

Base.intersect(Bsel1::AbstractBasisSelector, Bsel2::DownsetBasisSelector)  = intersect(Bsel2,Bsel1)
Base.intersect(Bsel1::AbstractBasisSelector, Bsel2::DownsetIntersection) = intersect(Bsel2,Bsel1)

function Base.intersect(Bsel1::DownsetIntersection, Bsel2::DownsetIntersection)
      Bsel = deepcopy(Bsel1)
      for b in Bsel2
            Bsel = intersect(Bsel, b)
      end
      return Bsel
end

function level(bb, Bsel::DownsetIntersection, basis::OneParticleBasis) 
   return maximum([ level(bb, bsel, basis)/maxlevel(bsel,basis) for bsel in Bsel.DBsels])
   #return maximum([ level(bb, bsel, basis)/maxlevel(length(bb),bsel,basis) for bsel in Bsel.DBsels])
end

function filter(bb, Bsel::DownsetIntersection, basis::OneParticleBasis) 
   return all([filter(bb, bsel, basis) for bsel in Bsel.DBsels]) && all([filter(bb, bsel, basis) for bsel in Bsel.ABsels])
end
=#
