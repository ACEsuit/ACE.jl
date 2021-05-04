
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "Basis Orthogonality"  begin

##

using Test
using ACE, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra
using ACE: TransformedJacobi, transform, transform_d, alloc_B, alloc_temp
using JuLIP: evaluate!

##

@info("Testing ortho-normality of Jacobi Polynomials")

α, β = 1 + rand(), 1 + rand()
N = 30
J = Jacobi(α, β, N)
Bj = alloc_B(J)
integrandJ = let J=J, B=Bj, α=α, β=β
   x ->  begin
            evaluate!(B, nothing, J, x)
            return B*B' * (1 - x)^α * (1+x)^β
         end
   end

Q = quadgk(integrandJ, -1, 1)[1]
println(@test round.(Q, digits=8) == Matrix(I, N+1, N+1))

##

@info("Testing ortho-normality of r-basis")

N = 20
rl, ru = 0.5, 3.0
fcut =  PolyCutoff2s(2, rl, ru)
trans = PolyTransform(2, 1.0)
P = TransformedJacobi(N, trans, fcut)
B = alloc_B(P)
tmp = alloc_temp(P)

integrand = let P = P, B = B, tmp = tmp
   r ->  begin
            evaluate!(B, tmp, P, r)
            return B * B' * abs(transform_d(P.trans, r))
         end
   end

Q = quadgk(integrand, rl, ru)[1]
println(@test round.(Q, digits=8) == Matrix(I, N+1, N+1))


##

@info("Testing (near-) orthonormality of r-basis via sampling")

# P = TransformedJacobi(...) from previous cell
Nsamples = 100_000

G = let
   G = zeros(length(P), length(P))
   for n = 1:Nsamples
      evaluate!(B, tmp, P, ACE.Utils.rand_radial(P))
      G += B * B'
   end
   G
end

println(@test cond(G) < 1.1)


##

@info("Testing (near-)orthonormality of Ylm-basis via sampling")

SH = ACE.SphericalHarmonics.SHBasis(5)

function gramian(SH::ACE.SphericalHarmonics.SHBasis, Nsamples=100_000)
   lenY = length(SH)
   G = zeros(ComplexF64, lenY, lenY)
   Y = alloc_B(SH)
   tmp = alloc_temp(SH)
   for n = 1:Nsamples
      evaluate!(Y, tmp, SH, ACE.Utils.rand_sphere())
      for i = 1:lenY, j = 1:lenY
         G[i,j] += Y[i] * Y[j]'
      end
   end
   return G / Nsamples
end

G = gramian(SH)
println(@test cond(G) < 1.1)


##

@info("Testing (near-)orthonormality of A-basis via sampling")

aceB = SHIPBasis( SparseSHIP(3, 5), trans, fcut )
function evalA(aceB, tmp, Rs)
   Zs = zeros(Int16, length(Rs))
   ACE.precompute_A!(tmp, aceB, Rs, Zs, 1)
   return tmp.A[1]
end

function A_gramian(aceB, Nsamples = 100_000)
   tmp = alloc_temp(aceB)
   lenA = length(tmp.A[1])
   G = zeros(ComplexF64, lenA, lenA)
   for n = 1:Nsamples
      R = ACE.rand_vec(aceB.J)
      A = evalA(aceB, tmp, [R])
      for i = 1:lenA, j = 1:lenA
         G[i,j] +=  A[i] * A[j]'
      end
   end
   return G
end

G = A_gramian(aceB)
println(@test cond(G) < 1.2)



end
