
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Basis Orthogonality"  begin

##

using Test
using SHIPs, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra, SHIPs.JacobiPolys
using SHIPs: TransformedJacobi, transform, transform_d, eval_basis!,
             alloc_B, alloc_temp


##

@info("Testing ortho-normality of Jacobi Polynomials")

α, β = 1 + rand(), 1 + rand()
N = 30
J = Jacobi(α, β, N)
Bj = alloc_B(J)
integrandJ = let J=J, B=Bj, α=α, β=β
   x ->  begin
            eval_basis!(B, nothing, J, x)
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
            eval_basis!(B, tmp, P, r)
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
      eval_basis!(B, tmp, P, SHIPs.Utils.rand_radial(P))
      G += B * B'
   end
   G
end

println(@test cond(G) < 1.1)


##

@info("Testing (near-)orthonormality of Ylm-basis via sampling")

SH = SHIPs.SphericalHarmonics.SHBasis(5)

function gramian(SH::SHIPs.SphericalHarmonics.SHBasis, Nsamples=100_000)
   lenY = length(SH)
   G = zeros(ComplexF64, lenY, lenY)
   Y = alloc_B(SH)
   tmp = alloc_temp(SH)
   for n = 1:Nsamples
      SHIPs.eval_basis!(Y, tmp, SH, SHIPs.Utils.rand_sphere())
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

shpB = SHIPBasis( SparseSHIP(3, 5), trans, fcut )
function evalA(shpB, tmp, Rs)
   Zs = zeros(Int16, length(Rs))
   SHIPs.precompute_A!(tmp, shpB, Rs, Zs, 1)
   return tmp.A[1]
end

function A_gramian(shpB, Nsamples = 100_000)
   tmp = alloc_temp(shpB)
   lenA = length(tmp.A[1])
   G = zeros(ComplexF64, lenA, lenA)
   for n = 1:Nsamples
      R = SHIPs.Utils.rand(shpB.J)
      A = evalA(shpB, tmp, [R])
      for i = 1:lenA, j = 1:lenA
         G[i,j] +=  A[i] * A[j]'
      end
   end
   return G
end

G = A_gramian(shpB)
println(@test cond(G) < 1.2)



end
