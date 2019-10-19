
using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ], zeros(Int16, N), 0
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

trans = PolyTransform(2, 1.0)
fcut = PolyCutoff2s(2, 0.5, 3.0)
BB = [ SHIPBasis(SparseSHIP(2, 20, wL=1.0), trans, fcut),
       SHIPBasis(SparseSHIP(3, 16, wL=1.5), trans, fcut),
       SHIPBasis(SparseSHIP(4, 12, wL=1.5), trans, fcut),
       SHIPBasis(SparseSHIP(5, 10, wL=1.5), trans, fcut) ]

Nat = 30
Rs, Zs, z0 = randR(Nat)
btmp = SHIPs.alloc_temp(BB[1])
@info("profile precomputation of A")
@btime SHIPs.precompute_A!($btmp, $(BB[1]), $Rs, $Zs, 1)

@info("profile ship-basis and fast-ship site energies")
for n = 2:5
   @info("  body-order $(n+1):")
   Rs, Zs, z0 = randR(Nat)
   B = BB[n-1]
   coeffs = randcoeffs(B)
   🚢 = SHIP(B, coeffs)
   b = SHIPs.alloc_B(B)
   btmp = SHIPs.alloc_temp(B, length(Rs))
   tmp = SHIPs.alloc_temp(🚢, length(Rs))
   @info("     evaluate a site energy:")
   print("         SHIPBasis: "); @btime SHIPs.eval_basis!($b, $btmp, $B, $Rs, $Zs, $z0)
   print("         SHIP     : "); @btime SHIPs.evaluate!($tmp, $🚢, $Rs, $Zs, $z0)

   dEs = zeros(JVecF, length(Rs))
   db = SHIPs.alloc_dB(B, length(Rs))
   dbtmp = SHIPs.alloc_temp_d(B, length(Rs))
   tmp = SHIPs.alloc_temp_d(🚢, length(Rs))

   @info("     site energy gradient:")
   print("         SHIPBasis: "); @btime SHIPs.eval_basis_d!($db, $dbtmp, $B, $Rs, $Zs, $z0)
   print("         SHIP     : "); @btime SHIPs.evaluate_d!($dEs, $tmp, $🚢, $Rs, $Zs, $z0)
end
