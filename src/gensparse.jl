
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# Auxiliary functions to generate sparse grid type stuff

# gensparse(N::Integer, deg, degfun, filter = _->true, TI = Int16) =
#       gensparse(N, ν -> ((degfun(ν) <= deg) && filter(ν)), TI)

gensparse(N::Integer;
         admissible = _->false,
         filter = _-> true,
         converter = identity,
         TI = Int16) =
   _gensparse(Val(N), admissible, filter, converter, TI)

function _gensparse(::Val{N},
                    admissible,
                    filter,
                    converter,
                    TI) where {N}
   @assert TI <: Integer
   @assert deg >= 0
   @assert N >= 1

   lastidx = 0
   ν = @MVector ones(TI, N)
   Nu = SVector{N, TI}[]

   while true
      # check whether the current ν tuple is admissible
      # the first condition is that its max index is small enough
      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down
      if admissible(ν)
         # ... then we add it to the stack  ...
         # (unless some filtering mechanism prevents it)
         if filter(ν)
            push!(Nu, converter(SVector(ν)))
         end
         # ... and increment it
         lastidx = N
         ν[lastidx] += 1
      else
         # we have overshot, e.g. degfun(ν) > deg; we must go back down, by
         # decreasing the index at which we increment
         if lastidx == 1
            break
         end
         ν[lastidx-1:end] .= ν[lastidx-1] + 1
         lastidx -= 1
      end
   end

   return Nu
end

end
