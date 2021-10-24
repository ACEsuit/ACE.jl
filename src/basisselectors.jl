
"""
`AbstractBasisSelector` : object specifying how a finite basis is selected from
the infinite space of symmetric polynomials.
"""
abstract type AbstractBasisSelector end

abstract type DownsetBasisSelector <: AbstractBasisSelector end

isadmissible(b, bsel::DownsetBasisSelector, basis::OneParticleBasis) = true

filter(b, bsel::DownsetBasisSelector, basis::OneParticleBasis) = true


"""
No constraints on the basis - this selects that largest possible basis
subject to additional constraints baked into the one-particle basis.
In practise this should be equivalent to a naive max-norm basis selection.
"""
struct MaxBasis <: DownsetBasisSelector
end

"""
`struct SimpleSparseBasis`:

The most basic form of a sparse basis selection, using the total degree.
Only the maximum correlation order and maximum degree may be specified.
This should only be used for testing.
"""
struct SimpleSparseBasis <: DownsetBasisSelector
   maxorder::Int
   maxlevel::Float64
end


# for a one-particle basis function
degree(b::NamedTuple, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      degree(b, basis)

# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
degree(bb, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      length(bb) == 0 ? 0 : sum( degree(b, basis) for b in bb )

isadmissible(b::NamedTuple, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      (degree(b, Bsel, basis) <= Bsel.maxlevel)

isadmissible(bb, Bsel::SimpleSparseBasis, basis::OneParticleBasis) =
      (degree(bb, Bsel, basis) <= Bsel.maxlevel) && length(bb) <= Bsel.maxorder

maxorder(Bsel::SimpleSparseBasis) = Bsel.maxorder



"""
`SparseBasis`: Super-type for sparse basis selection as sub-levelsets specified `levelset_fun` and corresponding (possibly order-dependent) 
levels provided in the dictionary `levels::Dict{Any, Float64}`. The function `levelset_fun` must be bounded from below by the degree function `degree` (i.e., `degree <= levelset_fun`).
Basis functions are selected in two steps. First, "admissible" basis specifications are generated as a sublevelset of the `degree` function using the implementation of the function `gensparse`. 
After that, basis functions that are not contained in the corresponding sub-levelset of `levelset_fun` are filtered out. 
"""

abstract type SparseBasis <: DownsetBasisSelector end

maxorder(Bsel::SparseBasis) = Bsel.maxorder

isadmissible(b::NamedTuple, Bsel::SparseBasis, basis::OneParticleBasis) =
      (degree(b, Bsel, basis) <= _maxlevel(Bsel, 0))

function isadmissible(bb, Bsel::SparseBasis, basis::OneParticleBasis)
   ord = length(bb)
   return ( degree(bb, Bsel, basis) <= _maxlevel(Bsel, ord)) && ord <= Bsel.maxorder
end

_maxlevel(Bsel::SparseBasis, ord::Integer) =
      haskey(Bsel.levels, ord) ? Bsel.levels[ord] : Bsel.levels["default"]

# for a one-particle basis function
degree(b::NamedTuple, Deg::SparseBasis, basis::OneParticleBasis) =
      degree(b, basis, Deg.weight)

# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
degree(bb, Bsel::SparseBasis, basis::OneParticleBasis)  =  (
      length(bb) == 0 ? 0.0
                      : norm( degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p ) )

# default implementation of `levelset_fun` assumes is `leveset_fun = degree`
levelset_fun(b::NamedTuple, Bsel::SparseBasis, basis::OneParticleBasis)  = degree(b, Bsel, basis) 

function levelset_fun(bb, Bsel::SparseBasis, basis::OneParticleBasis)
   return (length(bb) == 0 ? 0.0
                      : norm( levelset_fun.(bb, Ref(Bsel), Ref(basis)), Bsel.p ) )
end


filter(b::NamedTuple, bsel::DownsetBasisSelector, basis::OneParticleBasis) = isadmissible(b, Bsel, basis)

function filter(bb, Bsel::SparseBasis, basis::OneParticleBasis) 
   ord = length(bb)
   level_set_constr = (levelset_fun(bb, Bsel, basis) <= _maxlevel(Bsel, ord)) && ord <= Bsel.maxorder
   return level_set_constr
end

"""
`PNormSparseBasis`: simplest implementation of a `SparseBasis`.
"""
struct PNormSparseBasis <: SparseBasis
   maxorder::Int
   weight::Dict{Symbol, Float64}
   levels::Dict{Any, Float64}
   p::Float64
end

PNormSparseBasis(maxorder::Int;  p = 1, 
                                 weight = Dict(:l => 1.0, :n => 1.0), 
                                 default_maxlevel = nothing, 
                                 levels = Dict{Any, Float64}(),
                                 ) =   begin 
                                          if default_maxlevel != nothing  levels == Dict{Any, Float64}()
                                             PNormSparseBasis(maxorder, weight, Dict{Any, Float64}("default" => default_maxlevel), p)
                                          elseif default_maxlevel == nothing && levels != Dict{Any, Float64}()
                                             PNormSparseBasis(maxorder, weight, levels, p)
                                          else
                                             @error "Either both or neither optional arguments `levels` and `default_maxlevel` were provided. To avoid ambiguity ensure that exactly one of these arguments is provided."
                                          end
                                       end
"""
`ConstrainedSparseBasis`: similar to `SparseBasis` but allows in addition to specify
constraints via the function `constraint_is_satisfied`.
"""

abstract type ConstrainedSparseBasis <: SparseBasis end

constraint_is_satisfied(bb, Bsel::ConstrainedSparseBasis, basis::OneParticleBasis) = true

function filter(bb, Bsel::ConstrainedSparseBasis, basis::OneParticleBasis) 
   ord = length(bb)
   level_set_constr = (levelset_fun(bb, Bsel, basis) <= _maxlevel(Bsel, ord)) && ord <= Bsel.maxorder
   return level_set_constr && constraint_is_satisfied(bb, Bsel, basis)
end

"""
`CategorySparseBasis`: extension of `PNormSparseBasis` that implements a constraint on 
the correlation orders for within-category correlations, i.e., for each category `cat` contained in OneParticleBasis associated with 
   the symbol `isym`, it is required that the within-category correlation order `corr[cat]` satisfies
   ```
   minorder_dict[cat] <= corr[cat] <= maxorder_dict[cat].
   ```
"""
struct CategorySparseBasis <: ConstrainedSparseBasis
   maxorder::Int
   weight::Dict{Symbol, Float64}
   levels::Dict{Any, Float64}
   p::Float64
   isym::Symbol
   minorder_dict::Dict{Any, Int}
   maxorder_dict::Dict{Any, Int}
   weight_cat::Dict{Any, Float64} 
end



 #= 
 Ms: we should get rid of the arguments `categories` in the following constructor. 
 For that we must be able to access the basis element associated with `isym` in the product basis `basis`. 
 =#
CategorySparseBasis(maxorder::Int, isym::Symbol, categories::AbstractArray;
            p = 1, 
            weight = Dict{Symbol, Float64}(), 
            default_maxlevel = 10,
            levels = Dict{Any, Float64}(),
            minorder_dict = Dict{Any, Float64}(),
            maxorder_dict = Dict{Any, Float64}(),
            weight_cat = Dict(c => 1.0 for c in categories), 
            ) =   begin 
                     if default_maxlevel != nothing  levels == Dict{Any, Float64}() 
                        CategorySparseBasis(maxorder, weight, Dict{Any, Float64}("default" => default_maxlevel), p, isym, minorder_dict, maxorder_dict, weight_cat)
                     elseif default_maxlevel == nothing && levels != Dict{Any, Float64}()
                        CategorySparseBasis(maxorder, weight, levels, p, isym, minorder_dict, maxorder_dict, weight_cat)
                     else
                        @error "Either both or neither optional arguments `levels` and `default_maxlevel` were provided. To avoid ambiguity ensure that exactly one of these arguments is provided."
                     end
                  end
# for a one-particle basis function
levelset_fun(b::NamedTuple, Bsel::CategorySparseBasis, basis::OneParticleBasis) =
      degree(b, basis, Bsel.weight) * Bsel.weight_cat[getproperty(b, Bsel.isym)]

maxorder(Bsel::CategorySparseBasis, category) = Bsel.maxorder_dict[category]
minorder(Bsel::CategorySparseBasis, category) = Bsel.minorder_dict[category]

function constraint_is_satisfied(bb, Bsel::CategorySparseBasis, basis::OneParticleBasis) 
   cond_ord_cats_max = all([
      sum([ getproperty(b, Bsel.isym) == s for b in bb ]) <= maxorder(Bsel, s)
                              for s in keys(Bsel.maxorder_dict) ]  )
   cond_ord_cats_min = all([
      sum([ getproperty(b, Bsel.isym) == s for b in bb ]) >= minorder(Bsel, s)
                              for s in keys(Bsel.maxorder_dict) ]  )
   return cond_ord_cats_max && cond_ord_cats_min
end

