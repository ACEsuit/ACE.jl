
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module Jacobi

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
      B[n] = T( big(α^2 - β^2) * c2 / c1) )
      C[n] = T( big(-2*(n+α-1)*(n+β-1)*(2n+α+β)) / c1 )
   end
   return Jacobi(α, β, A, B, C)
end

function eval_basis!(P::AbstractVector, J::Jacobi, x::Number, N::Integer)
   @assert length(P) >= N+1
   @assert 0 <= N <= length(J.A)
   α, β = J.α, J.β
   P[1] = 1
   if N >= 1
      P[2] = (α+1) + 0.5 * (α+β+2) * (x-1)
   end
   for n = 2:N
      P[n+1] = (J.A[n] * x + J.B[n]) * P[n] + J.C[n] * P[n-1]
   end
   return P
end

function eval_grad!(P::AbstractVector, dP::AbstractVector,
                    J::Jacobi, x::Number, N::Integer)
   @assert length(P) >= N+1
   @assert 0 <= N <= length(J.A)
   α, β = J.α, J.β
   P[1] = 1
   dP[1] = 0
   if N >= 1
      P[2] = (α+1) + 0.5 * (α+β+2) * (x-1)
      dP[2] = 0.5 * (α+β+2)
   end
   for n = 2:N
      c1 = J.A[n] * x + J.B[n]
      c2 = J.C[n]
      P[n+1] = c1 * P[n] + c2 * P[n-1]
      dP[n+1] = J.A[n] * P[n] + c1 * dP[n] + J.C[n] * dP[n-1]
   end
   return P, dP
end


end
