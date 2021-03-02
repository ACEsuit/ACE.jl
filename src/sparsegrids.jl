# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# ----------- Auxiliary functions to generate sparse grid type stuff

"""
`function init1pspec!` : initialize the specification of the 1-particle basis,
generates all possible 1-p basis functions, sorted by degree.
"""
function init1pspec!(B1p::OneParticleBasis; maxdeg = Inf, Deg = B1p)
   syms = tuple(symbols(B1p)...)
   rgs = indexrange(B1p)
   lens = [ length(rgs[sym]) for sym in syms ]
   spec = NamedTuple{syms, NTuple{length(syms), Int}}[]
   for I in CartesianIndices(ntuple(i -> 1:lens[i], length(syms)))
      J = ntuple(i -> rgs[syms[i]][I.I[i]], length(syms))
      b = NamedTuple{syms}(J)
      # check whether valid
      if isadmissible(b, B1p)
         deg = degree(b, Deg)
         if deg <= maxdeg
            push!(spec, b)
         end
      end
   end
   sort!(spec, by = b -> degree(b, Deg))
   return set_spec!(B1p, spec)
end


gensparse(N::Integer, deg::Real; degfun = ν -> sum(ν), kwargs...) =
   gensparse(N; admissible = ν -> (degfun(ν) <= deg), kwargs...)

"""
`gensparse(...)` : utility function to generate high-dimensional sparse grids
which are downsets.

All arguments are keyword arguments (with defaults):
* `NU` : maximum correlation order
* `minvv = 0` : `minvv[i] gives the minimum value for `vv[i]`
* `maxvv = Inf` : `maxvv[i] gives the minimum value for `vv[i]`
* `tup2b = vv -> vv` :
* `admissible = _ -> false` : determines whether a tuple belongs to the downset
* `filter = _ -> true` : a callable object that returns true of tuple is to be kept and
false otherwise (whether or not it is part of the downset!) This is used, e.g.
to enfore conditions such as ∑ lₐ = even or |∑ mₐ| ≦ M
* `INT = Int` : integer type to be used
* `ordered = false` : whether only ordered tuples are produced; ordered tuples
correspond to  permutation-invariant basis functions
"""
gensparse(; NU::Integer = nothing,
            minvv = [0 for _=1:NU],
            maxvv = [Inf for _=1:NU],
            tup2b = vv -> vv,
            admissible = _-> false,
            filter = _-> true,
            INT = Int,
            ordered = false) =
      _gensparse(Val(NU), tup2b, admissible, filter, INT, ordered,
                 SVector(minvv...), SVector(maxvv...))

"""
`_gensparse` : function barrier for `gensparse`
"""
function _gensparse(::Val{NU}, tup2b, admissible, filter, INT, ordered,
                    minvv, maxvv) where {NU}
   @assert INT <: Integer

   lastidx = 0
   vv = @MVector zeros(INT, NU)
   for i = 1:NU; vv[i] = minvv[i]; end

   spec = SVector{NU, INT}[]
   orig_spec = SVector{NU, INT}[]

   if NU == 0
      if all(minvv .== 0) && isadmissible(vv) && filter(vv)
         push!(spec, SVector(vv))
      end
      return spec
   end

   while true
      # check whether the current vv tuple is admissible
      # the first condition is that its max index is small enough
      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down
      isadmissible = true
      if any(vv .> maxvv)
         isadmissible = false
      else
         b = tup2b(vv)
         isadmissible = admissible(b)
      end

      if isadmissible
         # ... then we add it to the stack  ...
         # (unless some filtering mechanism prevents it)
         if filter(b)
            push!(spec, SVector(vv))
            push!(orig_spec, copy(SVector(vv)))
         end
         # ... and increment it
         lastidx = NU
         vv[lastidx] += 1
      else
         if lastidx == 0
            error("""lastidx == 0 should never occur; this means that the
                     smallest basis function is already inadmissible and therefore
                     the basis is empty.""")
         end

         # we have overshot, e.g. degfun(vv) > deg; we must go back down, by
         # decreasing the index at which we increment
         if lastidx == 1
            # if we have gone all the way down to lastindex==1 and are still
            # inadmissible then this means we are done
            break
         end
         # reset
         vv[lastidx-1] += 1
         if ordered   #   ordered tuples (permutation symmetry)
            vv[lastidx:end] .= vv[lastidx-1]
         else         # unordered tuples (no permutation symmetry)
            vv[lastidx:end] .= 0
         end
         lastidx -= 1
      end
   end

   if ordered
      # sanity check, to make sure all is as intended...
      @assert all(issorted, orig_spec)
      @assert length(unique(orig_spec)) == length(orig_spec)
   end

   return identity.(spec)
end
