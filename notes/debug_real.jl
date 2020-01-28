
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



using PoSH, JuLIP
using JuLIP: evaluate
SH = PoSH.SphericalHarmonics

randR(N, J) = [ PoSH.Utils.rand(J) for n=1:N ], zeros(Int16, N), 0
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)
onecoeffs(B) = ones(length(B))

##

trans = PolyTransform(3, 1.0)
fcut = PolyCutoff2s(2, 0.5, 3.0)
B = SHIPBasis(SparseSHIP(3, 4, wL=1.0), trans, fcut)

# passed
izz = [1,1,1]; kk = [0,0,0]; ll = [0,1,1]; mm = [0,0,0]

# failed
# izz = [1,1,1]; kk = [0,0,0]; ll = [1,1,2]; mm = [0,1,-1]
izz = [1,1,1]; kk = [0,0,0]; ll = [0,1,1]; mm = [0,-1,1]

iAA = B.aalists[1][ (izz, kk, ll, mm) ]
coeffs = zeros(length(B))
# coeffs[iAA] = 1.0
# coeffs = onecoeffs(B)
# coeffs = randcoeffs(B)
ship = SHIP(B, coeffs)
ship.coeffs[1][iAA] = 1.0

rship = PoSH.convertc2r(ship)

##

println(); println()
for _=1:10
   N = 2
   Rs, Zs, z0 = randR(N, ship.J)
   cEs = evaluate(ship, Rs, Zs, z0)
   rEs = evaluate(rship, Rs, Zs, z0)
   @show cEs, rEs
   @show cEs ≈ rEs
end
