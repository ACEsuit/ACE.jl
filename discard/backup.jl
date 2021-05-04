
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


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
