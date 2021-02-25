

# ----------- Auxiliary functions to generate sparse grid type stuff

"""
`function init1pspec!` : initialize the specification of the 1-particle basis,
generates all possible 1-p basis functions, sorted by degree.
"""
function init1pspec!(B1p::OneParticleBasis)
   syms = tuple(symbols(B1p)...)
   rgs = indexrange(B1p)
   lens = [ length(rgs[sym]) for sym in syms ]
   spec = NamedTuple{syms, NTuple{length(syms), Int}}[]
   for I in CartesianIndices(ntuple(i -> 1:lens[i], length(syms)))
      J = ntuple(i -> rgs[syms[i]][I.I[i]], length(syms))
      b = NamedTuple{syms}(J)
      # check whether valid
      if isadmissible(b, B1p)
         push!(spec, b)
      end
   end
   return set_spec!(B1p, spec)
end



function gensparse( maxν::Integer,
                    maxdeg::Real,
                    B1p::OneParticleBasis,
                    D::AbstractDegree)
   # make the maximum correlation-order static
   valN = Val(maxν)
   #
end

gensparse(N::Integer, deg::Real; degfun = ν -> sum(ν), kwargs...) =
   gensparse(N; admissible = ν -> (degfun(ν) <= deg), kwargs...)

gensparse(N::Integer;
          admissible = _-> false,
          filter = _-> true,
          tup2b = ν -> SVector(ν),
          INT = Int,
          ordered = false,
          maxν = Inf) =
      _gensparse(Val(N), admissible, filter, tup2b, INT, ordered, maxν)

function _gensparse(::Val{N}, admissible, filter, tup2b, INT, ordered, maxν
                   ) where {N}
   @assert INT <: Integer

   lastidx = 0
   ν = @MVector zeros(INT, N)
   b = tup2b(ν)
   Nu = Vector{Any}(undef, 0)
   orig_Nu = []

   if N == 0
      push!(Nu, b)
      return Nu
   end

   while true
      # check whether the current ν tuple is admissible
      # the first condition is that its max index is small enough
      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down
      isadmissible = true
      if maximum(ν) > maxν
         isadmissible = false
      else
         b = tup2b(ν)
         isadmissible = admissible(b)
      end

      if isadmissible
         # ... then we add it to the stack  ...
         # (unless some filtering mechanism prevents it)
         if filter(b)
            push!(Nu, b)
            push!(orig_Nu, copy(ν))
         end
         # ... and increment it
         lastidx = N
         ν[lastidx] += 1
      else
         # we have overshot, e.g. degfun(ν) > deg; we must go back down, by
         # decreasing the index at which we increment
         if lastidx == 1
            # if we have gone all the way down to lastindex==1 and are still
            # inadmissible then this means we are done
            break
         end
         # reset
         ν[lastidx-1] += 1
         if ordered   #   ordered tuples (permutation symmetry)
            ν[lastidx:end] .= ν[lastidx-1]
         else         # unordered tuples (no permutation symmetry)
            ν[lastidx:end] .= 0
         end
         lastidx -= 1
      end
   end

   if ordered
      # @info("sanity test")
      @assert all(issorted, orig_Nu)
      @assert length(unique(orig_Nu)) == length(orig_Nu)
   end

   return identity.(Nu)
end
