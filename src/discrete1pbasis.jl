

import ACEbase: Discrete1pBasis
import ACE: evaluate, evaluate!

export Categorical1pBasis


# -------------------------

struct SList{N, T}
   list::SVector{N, T}

   function SList{N, T}(list::SVector{N, T})  where {N, T} 
      if isabstracttype(T)
         error("`SList` can only contain a single type")
      end
      return new(list)
   end
end

SList(list::AbstractArray) = SList(SVector(list...))
SList(args...) = SList(SVector(args...))
SList(list::SVector{N, T}) where {N, T} = SList{N, T}(list)

Base.length(list::SList) = length(list.list)
Base.rand(list::SList) = rand(list.list)

i2val(list::SList, i::Integer) = list.list[i]

Base.iterate(list::SList, args...) = iterate(list.list, args...)

function val2i(list::SList, val)
   for j = 1:length(list)
      if list.list[j] == val
         return j
      end
   end
   error("val = $val not found in this list")
end

write_dict(list::SList{N,T}) where {N, T} = 
      Dict( "__id__" => "ACE_SList", 
                 "T" => write_dict(T),
              "list" => list.list )

function read_dict(::Val{:ACE_SList}, D::Dict) 
   list = D["list"]
   T = read_dict(D["T"])
   svector = SVector{length(list), T}((T.(list))...)
   return SList(svector)
end



# -------------------------

@doc raw"""
`Categorical1pBasis` : defines the discrete 1p basis 
```math 
   \phi_q(u) = \delta(u - U_q),
```
where ``U_q, q = 1, \dots, Q`` are finitely many possible values that the 
variable ``u`` may take. Suppose, e.g., we allow the values `[:a, :b, :c]`, 
then 
```julia 
P = Categorical1pBasis([:a, :b, :c]; varsym = :u, idxsym = :q)
evaluate(P, State(u = :a))     # Bool[1, 0, 0]
evaluate(P, State(u = :b))     # Bool[0, 1, 0]
evaluate(P, State(u = :c))     # Bool[0, 0, 1]
```
If we evaluate it with an unknown state we get an error: 
```julia 
evaluate(P, State(u = :x))   
# Error : val = x not found in this list
```
"""
struct Categorical1pBasis{VSYM, ISYM, LEN, T} <: Discrete1pBasis{Bool}
   categories::SList{LEN, T}
   label::String 
end

_varsym(::Categorical1pBasis{VSYM, ISYM}) where {VSYM, ISYM} = VSYM
_isym(::Categorical1pBasis{VSYM, ISYM}) where {VSYM, ISYM} = ISYM

_val(X, B::Categorical1pBasis) = getproperty(X, _varsym(B))
_idx(b, B::Categorical1pBasis) = getproperty(b, _isym(B))

Base.length(B::Categorical1pBasis) = length(B.categories)

Categorical1pBasis(categories::AbstractArray; 
              varsym::Symbol = nothing, idxsym::Symbol = nothing, 
              label = "C$(idxsym)") = 
      Categorical1pBasis(categories, varsym, idxsym, label)

Categorical1pBasis(categories::AbstractArray, varsym::Symbol, isym::Symbol, label::String) = 
      Categorical1pBasis(SList(categories), varsym, isym, label)

Categorical1pBasis(categories::SList{LEN, T}, varsym::Symbol, isym::Symbol, label::String) where {LEN, T} = 
      Categorical1pBasis{varsym, isym, LEN, T}(categories, label)


function evaluate(basis::Categorical1pBasis, X::AbstractState)      
   A = Vector{Bool}(undef, length(basis))
   return evaluate!(A, basis, X)
end

function ACE.evaluate!(A, basis::Categorical1pBasis, X::AbstractState)
   fill!(A, false)
   A[val2i(basis.categories, _val(X, basis))] = true
   return A
end


symbols(basis::Categorical1pBasis) = [ _isym(basis), ]

indexrange(basis::Categorical1pBasis) = Dict( _isym(basis) => basis.categories.list )

isadmissible(b, basis::Categorical1pBasis) = (_idx(b, basis) in basis.categories)

get_index(B::Categorical1pBasis, b) = val2i(B.categories, _idx(b, B) )

degree(b, basis::Categorical1pBasis, args...) = 0

Base.rand(basis::Categorical1pBasis) = rand(basis.list)

function get_spec(basis::Categorical1pBasis, i)
   return NamedTuple{(_isym(basis),)}((i2val(basis.categories, i),))
end

get_spec(basis::Categorical1pBasis) = [ get_spec(basis, i) for i = 1:length(basis) ]


write_dict(B::Categorical1pBasis) = 
      Dict( "__id__" => "ACE_Categorical1pBasis", 
            "categories" => write_dict(B.categories), 
            "VSYM" => String(_varsym(B)), 
            "ISYM" => String(_isym(B)), 
            "label" => B.label)

read_dict(::Val{:ACE_Categorical1pBasis}, D::Dict)  = 
   Categorical1pBasis( read_dict(D["categories"]), 
                  Symbol(D["VSYM"]), Symbol(D["ISYM"]), 
                  D["label"] )
