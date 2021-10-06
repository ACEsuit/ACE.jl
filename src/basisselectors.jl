"""
`CategoryBasisSelector`: Simple basis selector enabling weighted degree
functions and varying degree for different correlation orders, and additional constraint
   to within-category
Need to add documentation; for now look at the code for `degree` to see how
the weight and degree dictionaries affect the definition of degree.
"""
struct CategoryBasisSelector <: DownsetBasisSelector
   maxorder::Int
   maxorder_dict::Dict{Symbol, Float64}
   isym::Symbol
   weight::Dict{Symbol, Float64}
   degree::Dict{Any, Float64}
   p::Float64
end



maxorder(Bsel::CategoryBasisSelector) = Bsel.maxorder
maxorder(Bsel::CategoryBasisSelector, category::Symbol, basis::Categorical1pBasis) = Bsel.maxorder_dict[category]
maxorder(Bsel::CategoryBasisSelector, category::Symbol, basis::Categorical1pBasis) = Bsel.maxorder_dict[category]

isadmissible(b::NamedTuple, Bsel::CategoryBasisSelector, basis::OneParticleBasis) =
      (degree(b, Bsel, basis) <= _maxdeg(Bsel, 0))

function isadmissible(bb, Bsel::CategoryBasisSelector, basis::OneParticleBasis)
   ord = length(bb)
   return (degree(bb, Bsel, basis) <= _maxdeg(Bsel, ord)) && ord <= maxorder(Bsel)
end

_maxdeg(Bsel::CategoryBasisSelector, ord::Integer) =
      haskey(Bsel.degree, ord) ? Bsel.degree[ord] : Bsel.degree["default"]

# for a one-particle basis function
degree(b::NamedTuple, Bsel::CategoryBasisSelector, basis::OneParticleBasis) =
      degree(b, basis, Bsel.weight)



# for an ν-correlation basis function
# in this case `bb` should be a Vector of NamedTuples
function degree(bb, Bsel::CategoryBasisSelector, basis::OneParticleBasis)
   if length(bb) == 0
      return 0.0
   elseif length(bb) > maxorder(Bsel) || any([sum([getproperty(b, Bsel.isym) == s for b in bb]) > maxorder(Bsel,basis, s) for s in keys(Bsel.maxorder) ])
      return Inf
   else
      return norm( degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p )
   end
end
