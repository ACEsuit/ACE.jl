

# -------------- Implementation of Product Basis

struct Product1PBasis{N, TB <: Tuple, T} <: OneParticleBasis{T}
   bases::TB
   spec::Vector{NTuple{N, Int}}
   _typeT::Type{T}
end

function Product1PBasis(bases)
   N = length(bases)
   T = promote_type(fltype.(bases)...)
   Product1PBasis( tuple(bases...), Vector{NTuple{N, Int}}(undef, 0), T )
end

_numb(b::Product1PBasis) = length(b.bases)

Base.length(basis::Product1PBasis) = length(basis.spec)

fltype(basis::Product1PBasis) = promote_type(fltype.(basis.bases)...)

alloc_temp(basis::Product1PBasis) =
      (
         B = alloc_B.(basis.bases),
         tmp = alloc_temp.(basis.bases)
      )

@generated function add_into_A!(A, tmp, basis::Product1PBasis{N}, Xj, Xi) where {N}
   quote
      Base.Cartesian.@nexprs $N i -> evaluate!(tmp.B[i], tmp.tmp[i], basis.bases[i], Xj, Xi)
      for (iA, ϕ) in enumerate(basis.spec)
         t = one(eltype(A))
         Base.Cartesian.@nexprs $N i -> (t *= tmp.B[i][ϕ[i]])
         A[iA] += t
      end
      return nothing
   end
end

symbols(basis::Product1PBasis) = union( symbols.(basis.bases)... )

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

function set_spec!(basis::Product1PBasis, spec)
   basis.spec = spec
   return basis
end 
