

# -------------- Implementation of Product Basis

struct Product1pBasis{NB, TB <: Tuple, NSYM, SYMS, T} <: OneParticleBasis{T}
   bases::TB
   spec::Vector{NamedTuple{SYMS, NTuple{NSYM, Int}}}
   indices::Vector{NTuple{NB, Int}}
   _typeT::Type{T}
end

function Product1pBasis(bases;
                        SYMS = _symbols_prod(bases),
                        T = promote_type(fltype.(bases)...) )
   # TODO: discuss whether to construct an optimal ordering, e.g.
   #       should the discrete bases come first once we implement the
   #       "strongzero" method?
   NSYM = length(SYMS)
   NB = length(bases)
   Product1pBasis( tuple(bases...),
                   NamedTuple{SYMS, NTuple{NSYM, Int}}[],
                   NTuple{NB, Int}[],
                   T )
end


import Base.*
*(B1::OneParticleBasis, B2::OneParticleBasis) =
      Product1pBasis((B1, B2))
*(B1::Product1pBasis, B2::OneParticleBasis) =
      Product1pBasis((B1.bases..., B2))
*(B1::OneParticleBasis, B2::Product1pBasis) =
      Product1pBasis((B1, B2.bases...))
*(B1::Product1pBasis, B2::Product1pBasis) =
      Product1pBasis((B1.bases..., B2.bases...))


_numb(b::Product1pBasis{NB}) where {NB} = NB

Base.length(basis::Product1pBasis) = length(basis.spec)

fltype(basis::Product1pBasis) = promote_type(fltype.(basis.bases)...)

alloc_temp(basis::Product1pBasis, args...) =
      (
         B = alloc_B.(basis.bases),
         tmp = alloc_temp.(basis.bases)
      )

alloc_temp_d(basis::Product1pBasis, args...) =
      (
         B = alloc_B.(basis.bases),
         tmp = alloc_temp.(basis.bases),
         dB = alloc_dB.(basis.bases),
         tmpd = alloc_temp_d.(basis.bases)
      )


function gradtype(basis::Product1pBasis)
   # get the types of the sub-bases ...
   B = alloc_B.(basis.bases)
   dB = alloc_dB.(basis.bases)
   # ... do some artificial arithmetic  ...
   x = dB[1][1] * prod( B[a][1] for a = 2:length(B) )
   for a = 2:length(B)
      y = dB[a][1]
      for b = 1:length(B)
         b == a && continue
         y *= B[b][1]
      end
      x += y
   end
   # ... to find out the end-result of the gradient calculation
   return typeof(x)
end

@generated function add_into_A!(A, tmp, basis::Product1pBasis{NB}, X) where {NB}
   quote
      Base.Cartesian.@nexprs $NB i -> evaluate!(tmp.B[i], tmp.tmp[i], basis.bases[i], X)
      for (iA, ϕ) in enumerate(basis.indices)
         t = one(eltype(A))
         Base.Cartesian.@nexprs $NB i -> (t *= tmp.B[i][ϕ[i]])
         A[iA] += t
      end
      return nothing
   end
end



@generated function add_into_A_dA!(A, dA, tmpd, basis::Product1pBasis{NB}, X
                                   ) where {NB}
   quote
      Base.Cartesian.@nexprs($NB, i -> begin   # for i = 1:NB
         evaluate_ed!(tmpd.B[i], tmpd.dB[i], tmpd.tmpd[i], basis.bases[i], X)
      end)
      for (iA, ϕ) in enumerate(basis.indices)
         # evaluate A
         t = one(eltype(A))
         Base.Cartesian.@nexprs($NB, i -> begin   # for i = 1:NB
            t *= tmpd.B[i][ϕ[i]]
         end)
         A[iA] += t

         # evaluate dA
         dA[iA] = zero(eltype(dA))
         Base.Cartesian.@nexprs($NB, a -> begin  # for a = 1:NB
            dt = tmpd.dB[a][ϕ[a]]
            Base.Cartesian.@nexprs($NB, b -> begin  # for b = 1:NB
               if b != a
                  dt *= tmpd.B[b][ϕ[b]]
               end
            end)
            dA[iA] += dt
         end)
      end
      return nothing
   end
end



_symbols_prod(bases) = tuple(union( symbols.(bases)... )...)

symbols(basis::Product1pBasis{NB, TB, NSYM, SYMS}
            ) where {NB, TB, NSYM, SYMS} = SYMS

function indexrange(basis::Product1pBasis)
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

isadmissible(b, basis::Product1pBasis) = all(isadmissible.(Ref(b), basis.bases))

function set_spec!(basis::Product1pBasis{NB}, spec) where {NB}
   empty!(basis.spec)
   append!(basis.spec, spec)
   empty!(basis.indices)
   for b in basis.spec
      inds = ntuple(i -> get_index(basis.bases[i], b), NB)
      push!(basis.indices, inds)
   end
   return basis
end

get_spec(basis::Product1pBasis) = basis.spec

get_spec(basis::Product1pBasis, i::Integer) = basis.spec[i]

degree(b, basis::Product1pBasis) = sum( degree(b, B) for B in basis.bases )



# TODO: this looks like a horrible hack ...
function rand_radial(basis::Product1pBasis)
   for B in basis.bases
      if B isa ScalarACEBasis
         return rand_radial(B)
      end
   end
   return nothing
end
