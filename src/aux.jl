
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



# -----------------------------------
# iterating over an m collection
# -----------------------------------

_mvec(::CartesianIndex{0}) = SVector(IntS(0))

_mvec(mpre::CartesianIndex) = SVector(Tuple(mpre)..., - sum(Tuple(mpre)))

"""
Given an l-vector `ll` iterate over all combinations of `mm` vectors  of
the same length such that `sum(mm) == 0`
"""
_mrange(ll) = (
                  _mvec(mpre) for mpre in CartesianIndices(
                                ntuple(i -> -ll[i]:ll[i], length(ll)-1) )
              )


"""
`@generated function nfcalls(::Val{N}, f)`

Effectively generates a loop of functions calls, but fully unrolled and
therefore type-stable:
```{julia}
f(Val(1))
f(Val(2))
f(Val(3))
# ...
f(Val(N))
```
"""
@generated function nfcalls(::Val{N}, f) where {N}
   code = Expr[]
   for n = 1:N
      push!(code, :(f(Val($n))))
   end
   quote
      $(Expr(:block, code...))
      return nothing
   end
end


"""
`@generated function valnmapreduce(::Val{N}, v, f)`

Generates a map-reduce like code, with fully unrolled loop which makes this
type-stable,
```{julia}
begin
   v += f(Val(1))
   v += f(Val(2))
   # ...
   v += f(Val(N))
   return v
end
```
"""
@generated function valnmapreduce(::Val{N}, v, f) where {N}
   code = Expr[]
   for n = 1:N
      push!(code, :(v += f(Val($n))))
   end
   quote
      $(Expr(:block, code...))
      return v
   end
end
