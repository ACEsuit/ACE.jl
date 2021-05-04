
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



# TODO: QuadGK is used to "hack" normalised Jacobi Polynomials.
#       to make sure they are orthonormal, not just orthogonal,
#       but this should be done properly...
#       or - indeed - move all this to a more general Pair potential
#       with arbitrary orthogonality - measure

using QuadGK

import JuLIP: evaluate,
              evaluate!,
              evaluate_d!,
              read_dict,
              write_dict

import JuLIP.MLIPs: alloc_B, alloc_dB, IPBasis

import Base.==

export Jacobi, Chebyshev


Chebyshev(N, args...; kwargs...) = Jacobi(0.5, 0.5, N, args...; kwargs...)


"""
`Jacobi{T} : ` represents the basis of Jacobi polynomials
parameterised by α, β up to some fixed maximum degree. Recall that Jacobi
polynomials are orthogonal on [-1,1] w.r.t. the weight
w(x) = (1-x)^α (1+x)^β.

### Constructor:
```
Jacobi(α, β, N)   # N = max degree
```

### Evaluate basis and derivatives:
```
x = 2*(rand() - 0.5)
P = zeros(N)
evaluate!(P, J, x)
dP = zeros(N)
evaluate_d!(P, dP, J, x)   # evaluates both P, dP
```

### Notes

`Jacobi(...)` precomputes the recursion coefficients using arbitrary
precision arithmetic, then stores them as `Vector{Float64}`. The recursion
is then given by
```
P_{n} = (A[n] * x + B[n]) * P_{n-1} + C[n] * P_{n-2}
```
"""
struct Jacobi{T} <: IPBasis
   α::T
   β::T
   A::Vector{T}
   B::Vector{T}
   C::Vector{T}
   nrm::Vector{T}
end

==(J1::Jacobi, J2::Jacobi) = (
      (J1.α == J2.α) && (J1.β == J2.β) && (length(J1) == length(J2))
   )

write_dict(J::Jacobi{T}) where {T} =
      Dict( "__id__" => "ACE_Jacobi",
            "T" => write_dict(T),
            "alpha" => J.alpha,
            "beta" => J.beta,
            "N" => length(J.A),
            "normalise" => !isempty(J.nrm) )

read_dict(::Val{SHIPs_Jacobi}, D::Dict) = read_dict(Val{ACE_Jacobi}(), D)

read_dict(::Val{:ACE_Jacobi}, D::Dict; T = read_dict(D["T"])) =
      Jacobi(T(D["alpha"]), T(D["beta"]), D["N"], T = T;
             normalise = D["normalise"])


function Jacobi(α, β, N, T=Float64; normalise=true)
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
   J = Jacobi(T(α), T(β), A, B, C, T[])
   if normalise
      integrand = x -> evaluate(J, x).^2 * ((1-x)^α * (1+x)^β)
      nrm2 = quadgk(integrand, -1.0, 1.0)[1]
      J = Jacobi(T(α), T(β), A, B, C, nrm2.^(-0.5))
   end
   return J
end


Base.length(J::Jacobi) = maxdegree(J) + 1
maxdegree(J::Jacobi) = length(J.A)
alloc_B(J::Jacobi{T}, args...) where {T} = zeros(T, length(J))
alloc_dB(J::Jacobi{T}, args...) where {T} = zeros(T, length(J))

function evaluate!(P::AbstractVector, tmp, J::Jacobi, x)
   N = maxdegree(J)
   @assert length(P) >= N + 1
   @assert N+1 <= length(P)
   α, β = J.α, J.β
   P[1] = 1
   if N > 0
      #       (A * x + B) * P[1]
      #       A = 0.5 * (α+β+2)
      #       B = (α+1) - 0.5 * (α+β+2)
      P[2] = (α+1) + 0.5 * (α+β+2) * (x-1)
      if N > 1
         # 3-pt recursion
         @inbounds for n = 2:N
            P[n+1] = (J.A[n] * x + J.B[n]) * P[n] + J.C[n] * P[n-1]
         end
      end
   end
   # normalise
   if !isempty(J.nrm)  # if we want an orthonormal basis
      P .= P .* J.nrm
   end
   return P
end


function evaluate_d!(P::AbstractVector, dP::AbstractVector, tmp,
                    J::Jacobi, x::Number)
   N = maxdegree(J)
   @assert length(P) >= N+1
   @assert length(dP) >= N+1
   @assert N+1 <= min(length(P), length(dP))
   α, β = J.α, J.β
   P[1] = 1
   dP[1] = 0
   if N > 0
      P[2] = (α+1) + 0.5 * (α+β+2) * (x-1)
      dP[2] = 0.5 * (α+β+2)
      if N > 1
         @inbounds for n = 2:N
            c1 = J.A[n] * x + J.B[n]
            c2 = J.C[n]
            P[n+1] = c1 * P[n] + c2 * P[n-1]
            dP[n+1] = J.A[n] * P[n] + c1 * dP[n] + J.C[n] * dP[n-1]
         end
      end
   end
   if !isempty(J.nrm)  # if we want an orthonormal basis
      P .= P .* J.nrm
      dP .= dP .* J.nrm
   end
   # return P, dP
   return dP
end
