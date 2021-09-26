

import ACEbase: Discrete1pBasis

export Onehot1pBasis


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

"""
`Onehot1pBasis` : todo write docs 
"""
struct Onehot1pBasis{VSYM, ISYM, LEN, T} <: Discrete1pBasis{LEN}
   categories::SList{LEN, T}
end

_varsym(::Onehot1pBasis{VSYM, ISYM}) where {VSYM, ISYM} = VSYM
_isym(::Onehot1pBasis{VSYM, ISYM}) where {VSYM, ISYM} = ISYM

_val(X, B::Onehot1pBasis) = getproperty(X, _varsym(B))
_idx(b, B::Onehot1pBasis) = getproperty(b, _isym(B))

Base.length(B::Onehot1pBasis) = length(B.categories)

Onehot1pBasis(categories::AbstractArray, varsym::Symbol, isym::Symbol) = 
      Onehot1pBasis(SList(categories), varsym, isym)

Onehot1pBasis(categories::SList{LEN, T}, varsym::Symbol, isym::Symbol) where {LEN, T} = 
      Onehot1pBasis{varsym, isym, LEN, T}(categories)

function ACE.evaluate!(A, basis::Onehot1pBasis, X::AbstractState)
   fill!(A, false)
   A[val2i(basis.categories, _val(X, basis))] = true
   return A
end

ACE.valtype(::Onehot1pBasis, args...) = Bool

symbols(basis::Onehot1pBasis) = [ _isym(basis), ]

indexrange(basis::Onehot1pBasis) = Dict( _isym(basis) => basis.categories.list )

isadmissible(b, basis::Onehot1pBasis) = (_idx(b, basis) in basis.categories)

degree(b, basis::Onehot1pBasis, args...) = 0

Base.rand(basis::Onehot1pBasis) = rand(basis.list)


write_dict(B::Onehot1pBasis) = 
      Dict( "__id__" => "ACE_Onehot1pBasis", 
            "categories" => write_dict(B.categories), 
            "VSYM" => String(_varsym(B)), 
            "ISYM" => String(_isym(B)))

read_dict(::Val{:ACE_Onehot1pBasis}, D::Dict)  = 
   Onehot1pBasis( read_dict(D["categories"]), 
                  Symbol(D["VSYM"]), Symbol(D["ISYM"]) )
