
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



using StaticArrays
using LinearAlgebra: norm, dot


# ------------------------------------------------------------
# miscellaneous

"""
a simple utility function to check whether two objects are equal
"""
_allfieldsequal(x1, x2) =
      all( getfield(x1, sym) == getfield(x2, sym)
           for sym in union(fieldnames(typeof(x1)), fieldnames(typeof(x2))) )


# ----------- Auxiliary functions to generate sparse grid type stuff
# -> could go into Utils?

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
