
module JacobiPolys

# TODO: QuadGK is used to "hack" normalised Jacobi Polynomials.
#       to make sure they are orthonormal, not just orthogonal,
#       but this should be done properly...
#       or - indeed - move all this to a more general Pair potential
#       with arbitrary orthogonality - measure

using QuadGK

import PolyPairPots: alloc_B, alloc_dB

import JuLIP.Potentials: evaluate, evaluate!, evaluate_d!

import Base.==
export Jacobi

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
P = zeros(length(J))
evaluate!(P, J, x, N)
dP = zeros(length(J))
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
struct Jacobi{T}
   α::T
   β::T
   A::Vector{T}
   B::Vector{T}
   C::Vector{T}
   nrm::Vector{T}
   skip0::Bool
end

==(J1::Jacobi, J2::Jacobi) = (
      (J1.α == J2.α) && (J1.β == J2.β) && (length(J1) == length(J2))
   )


function Jacobi(α, β, N, T=Float64; normalise=true, skip0=false)
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
   J = Jacobi(T(α), T(β), A, B, C, T[], skip0)
   if normalise
      integrand = x -> evaluate(J, x).^2 * ((1-x)^α * (1+x)^β)
      nrm2 = quadgk(integrand, -1.0, 1.0)[1]
      J = Jacobi(T(α), T(β), A, B, C, nrm2.^(-0.5), skip0)
   end
   return J
end

Base.length(J::Jacobi) = J.skip0 ? maxdegree(J) : maxdegree(J) + 1
maxdegree(J::Jacobi) = length(J.A)
alloc_B(J::Jacobi{T}, args...) where {T} = zeros(T, length(J))
alloc_dB(J::Jacobi{T}, args...) where {T} = zeros(T, length(J))

evaluate(J::Jacobi, x) where {T} = evaluate!( alloc_B(J), nothing, J, x )
evaluate_d(J::Jacobi, x) where {T} = evaluate!( alloc_B(J), alloc_dB(J), nothing, J, x )

function evaluate!(P::AbstractVector, tmp, J::Jacobi, x)
   N = maxdegree(J) #::Integer = length(P)-1
   @assert (length(P) >= N + 1 - J.skip0)
   @assert 2 <= N <= maxdegree(J)
   α, β = J.α, J.β
   @inbounds begin
      if J.skip0
         i0 = -1
      else
         P[1] = 1
         i0 = 0
      end
      P[i0+2] = (α+1) + 0.5 * (α+β+2) * (x-1)
      for n = 2:N
         P[i0+n+1] = (J.A[n] * x + J.B[n]) * P[i0+n] + J.C[n] * P[i0+n-1]
      end
   end
   if !isempty(J.nrm)  # if we want an orthonormal basis
      P .= P .* J.nrm
   end
   return P
end


function evaluate_d!(P::AbstractVector, dP::AbstractVector, tmp,
                     J::Jacobi, x::Number)
   N = maxdegree(J) #::Integer = length(P)-1
   @assert length(P) >= N+1
   @assert length(dP) >= N+1
   @assert 2 <= N <= maxdegree(J)
   α, β = J.α, J.β
   @inbounds begin
      if J.skip0
         i0 = -1
      else
         P[1] = 1
         dP[1] = 0
         i0 = 0
      end
      P[i0+2] = (α+1) + 0.5 * (α+β+2) * (x-1)
      dP[i0+2] = 0.5 * (α+β+2)
      for n = 2:N
         c1 = J.A[n] * x + J.B[n]
         c2 = J.C[n]
         P[i0+n+1] = c1 * P[i0+n] + c2 * P[i0+n-1]
         dP[i0+n+1] = J.A[n] * P[i0+n] + c1 * dP[n] + J.C[n] * dP[i0+n-1]
      end
   end # @inbounds
   if !isempty(J.nrm)  # if we want an orthonormal basis
      P .= P .* J.nrm
      dP .= dP .* J.nrm
   end
   return dP
end


end
