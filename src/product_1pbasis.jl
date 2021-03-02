# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# -------------- Implementation of Product Basis

struct Product1PBasis{NB, TB <: Tuple, NSYM, SYMS, T} <: OneParticleBasis{T}
   bases::TB
   spec::Vector{NamedTuple{SYMS, NTuple{NSYM, Int}}}
   indices::Vector{NTuple{NB, Int}}
   _typeT::Type{T}
end

function Product1PBasis(bases;
                        SYMS = _symbols_prod(bases),
                        T = promote_type(fltype.(bases)...) )
   NSYM = length(SYMS)
   NB = length(bases)
   Product1PBasis( tuple(bases...),
                   NamedTuple{SYMS, NTuple{NSYM, Int}}[],
                   NTuple{NB, Int}[],
                   T )
end

_numb(b::Product1PBasis{NB}) where {NB} = NB

Base.length(basis::Product1PBasis) = length(basis.spec)

fltype(basis::Product1PBasis) = promote_type(fltype.(basis.bases)...)

alloc_temp(basis::Product1PBasis) =
      (
         B = alloc_B.(basis.bases),
         tmp = alloc_temp.(basis.bases)
      )

@generated function add_into_A!(A, tmp, basis::Product1PBasis{NB}, Xj, Xi) where {NB}
   quote
      Base.Cartesian.@nexprs $NB i -> evaluate!(tmp.B[i], tmp.tmp[i], basis.bases[i], Xj, Xi)
      for (iA, ϕ) in enumerate(basis.indices)
         t = one(eltype(A))
         Base.Cartesian.@nexprs $NB i -> (t *= tmp.B[i][ϕ[i]])
         A[iA] += t
      end
      return nothing
   end
end

_symbols_prod(bases) = tuple(union( symbols.(bases)... )...)

symbols(basis::Product1PBasis{NB, TB, NSYM, SYMS}
            ) where {NB, TB, NSYM, SYMS} = SYMS

function indexrange(basis::Product1PBasis)
   rg = Dict{Symbol, Vector{Int}}()
   allsyms = symbols(basis)
   for sym in allsyms
      rg[sym] = Int[]
   end
   for b in basis.bases
      rgb = indexrange(b)
      for sym in allsyms
         if haskey(rgb, sym)
            rg[sym] = union(rg[sym], rgb[sym])
         end
      end
   end

   # HACK: fix the m range based on the maximal l-range
   #       this needs to be suitably generalised if we have multiple
   #       (l, m) pairs, e.g. (l1, m1), (l2, m2)
   if haskey(rg, :m)
      maxl = maximum(rg[:l])
      rg[:m] = collect(-maxl:maxl)
   end

   return rg
end

isadmissible(b, basis::Product1PBasis) = all(isadmissible.(Ref(b), basis.bases))

function set_spec!(basis::Product1PBasis{NB}, spec) where {NB}
   empty!(basis.spec)
   append!(basis.spec, spec)
   empty!(basis.indices)
   for b in basis.spec
      inds = ntuple(i -> get_index(basis.bases[i], b), NB)
      push!(basis.indices, inds)
   end
   return basis
end

get_spec(basis::Product1PBasis) = basis.spec

get_spec(basis::Product1PBasis, i::Integer) = basis.spec[i]

degree(b, basis::Product1PBasis) = sum( degree(b, B) for B in basis.bases )
