
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module JacobiPolys

export Jacobi, eval_basis, eval_basis!, eval_grad, eval_grad!

"""
`Jacobi{T} : ` represents the basis of Jacobi polynomials
parameterised by α, β up to some fixed maximum degree.

Constructor:
```
Jacobi(α, β, N)   # N = max degree
```

Evaluate basis and derivatives:
```
x = 2*(rand() - 0.5)
P = zeros(N)
eval_basis!(P, J, x, N)
dP = zeros(N)
eval_grad!(P, dP, J, x, N)   # evaluates both P, dP
```
"""
# the recursion is then given by
#   P_{n} = (A[n] * x + B[n]) * P_{n-1} + C[n] * P_{n-2}
struct Jacobi{T}
   α::T
   β::T
   A::Vector{T}
   B::Vector{T}
   C::Vector{T}
end

function Jacobi(α, β, N, T=Float64)
   # precompute the recursion coefficients
   A = zeros(T, N)
   B = zeros(T, N)
   C = zeros(T, N)
   for n = 2:N
      c1 = big(2*n*(n+α+β)*(2*n+α+β-2))
      c2 = big(2*n+α+β-1)
      A[n] = T( big(2*n+α+β)*big(2*n+α+β-2)*c2 / c1 )
      B[n] = T( big(α^2 - β^2) * c2 / c1 )
      C[n] = T( big(-2*(n+α-1)*(n+β-1)*(2n+α+β)) / c1 )
   end
   return Jacobi(α, β, A, B, C)
end

function eval_basis!(P::AbstractVector, J::Jacobi, x,
                     N::Integer = length(P)-1 )
   @assert length(P) >= N+1
   @assert 0 <= N <= length(J.A)
   α, β = J.α, J.β
   @inbounds P[1] = 1
   @inbounds if N >= 1; P[2] = (α+1) + 0.5 * (α+β+2) * (x-1); end
   for n = 2:N
      @inbounds P[n+1] = (J.A[n] * x + J.B[n]) * P[n] + J.C[n] * P[n-1]
   end
   return P
end

eval_basis(J::Jacobi, x::Number, N::Integer, T=Float64) =
      eval_basis!(zeros(T, N+1), J, x, N)

eval_basis(J, x::Number, T=Float64) =
      eval_basis(J, x, length(J.A), T)


function eval_grad!(P::AbstractVector, dP::AbstractVector,
                    J::Jacobi, x::Number, N::Integer = length(P)-1)
   @assert length(P) >= N+1
   @assert length(dP) >= N+1
   @assert 0 <= N <= length(J.A)
   α, β = J.α, J.β
   @inbounds P[1] = 1
   @inbounds dP[1] = 0
   if N >= 1
      @inbounds P[2] = (α+1) + 0.5 * (α+β+2) * (x-1)
      @inbounds dP[2] = 0.5 * (α+β+2)
   end
   for n = 2:N
      @inbounds c1 = J.A[n] * x + J.B[n]
      @inbounds c2 = J.C[n]
      @inbounds P[n+1] = c1 * P[n] + c2 * P[n-1]
      @inbounds dP[n+1] = J.A[n] * P[n] + c1 * dP[n] + J.C[n] * dP[n-1]
   end
   return P, dP
end

eval_grad(J::Jacobi, x::Number, N::Integer, T=Float64) =
      eval_grad!(zeros(T, N+1), zeros(T, N+1), J, x, N)

eval_grad(J, x::Number, T=Float64) =
      eval_grad(J, x, length(J.A), T)



# Transformed Jacobi Polynomials
# ------------------------------



end
