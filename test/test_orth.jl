
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Basis Orthogonality"

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


end
