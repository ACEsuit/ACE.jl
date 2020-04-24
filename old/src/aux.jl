
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using StaticArrays
using SparseArrays: SparseMatrixCSC

# -----------------------------------
# iterating over an m collection
# -----------------------------------

_mvec(::CartesianIndex{0}) = SVector(IntS(0))

_mvec(mpre::CartesianIndex) = SVector(Tuple(mpre)..., - sum(Tuple(mpre)))

struct MRange{N, TI, T2}
   ll::SVector{N, TI}
   cartrg::T2
end

Base.length(mr::MRange) = sum(_->1, _mrange(mr.ll))

"""
Given an l-vector `ll` iterate over all combinations of `mm` vectors  of
the same length such that `sum(mm) == 0`
"""
_mrange(ll) = MRange(ll, Iterators.Stateful(
                        CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)-1))))

function Base.iterate(mr::MRange{1, TI}, args...) where {TI}
   if isempty(mr.cartrg)
      return nothing
   end
   while !isempty(mr.cartrg)
      popfirst!(mr.cartrg)
   end
   return SVector{1, TI}(0), nothing
end

function Base.iterate(mr::MRange, args...)
   while true
      if isempty(mr.cartrg)
         return nothing
      end
      mpre = popfirst!(mr.cartrg)
      if abs(sum(mpre.I)) <= mr.ll[end]
         return _mvec(mpre), nothing
      end
   end
   error("we should never be here")
end



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



# ----------------------------------------------------------------------
# sparse matrix multiplication with weaker type restrictions


function _my_mul!(C::AbstractVector, A::SparseMatrixCSC, B::AbstractVector)
   A.n == length(B) || throw(DimensionMismatch())
   A.m == length(C) || throw(DimensionMismatch())
   nzv = A.nzval
   rv = A.rowval
   fill!(C, zero(eltype(C)))
   @inbounds for col = 1:A.n
      b = B[col]
      for j = A.colptr[col]:(A.colptr[col + 1] - 1)
         C[rv[j]] += nzv[j] * b
      end
   end
   return C
end




##
# ----------- Auxiliary functions to generate sparse grid type stuff


gensparse(N::Integer, deg::Integer; degfun = ν -> sum(ν), kwargs...) =
   gensparse(N; admissible = (degfun(ν) <= deg), kwargs...)

gensparse(N::Integer;
          admissible = _-> false,
          filter = _-> true,
          INT = Int16,
          ordered = false) =
      _gensparse(Val(N), admissible, filter, INT, ordered)

function _gensparse(::Val{N}, admissible, filter, INT, ordered) where {N}
   @assert INT <: Integer

   lastidx = 0
   ν = @MVector zeros(INT, N)
   Nu = SVector{N, INT}[]

   if N == 0
      push!(Nu, SVector{N, INT}())
      return Nu
   end

   while true
      # check whether the current ν tuple is admissible
      # the first condition is that its max index is small enough
      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down
      if admissible(ν)
         # ... then we add it to the stack  ...
         # (unless some filtering mechanism prevents it)
         if filter(ν)
            push!(Nu, SVector(ν))
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

   return Nu
end
