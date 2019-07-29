
using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ]
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

trans = PolyTransform(2, 1.0)
fcut = PolyCutoff2s(2, 0.5, 3.0)
BB = [ SHIPBasis(SparseSHIP(20, 1.0), 2, trans, fcut),
       SHIPBasis(SparseSHIP(16, 1.5), 3, trans, fcut),
       SHIPBasis(SparseSHIP(12, 1.5), 4, trans, fcut),
       SHIPBasis(SparseSHIP(10, 1.5), 5, trans, fcut) ]

Nat = 30
Rs = randR(Nat)
btmp = SHIPs.alloc_temp(BB[1])
@info("profile precomputation of A")
@btime SHIPs.precompute_A!($btmp, $(BB[1]), $Rs)

@info("profile ship-basis and fast-ship site energies")
for n = 2:5
   @info("  body-order $(n+1):")
   Rs = randR(Nat)
   B = BB[n-1]
   coeffs = randcoeffs(B)
   🚢 = SHIP(B, coeffs)
   b = SHIPs.alloc_B(B)
   btmp = SHIPs.alloc_temp(B, length(Rs))
   tmp = SHIPs.alloc_temp(🚢, length(Rs))
   @info("     evaluate a site energy:")
   print("         SHIPBasis: "); @btime SHIPs.eval_basis!($b, $B, $Rs, $tmp)
   print("         SHIP     : "); @btime SHIPs.evaluate!($tmp, $🚢, $Rs)

   tmp = SHIPs.alloc_temp_d(🚢, Rs)
   dEs = zeros(JVecF, length(Rs))
   db = SHIPs.alloc_dB(B, length(Rs))
   dbtmp = SHIPs.alloc_temp_d(B, length(Rs))

   @info("     site energy gradient:")
   store = SHIPs.alloc_temp_d(🚢, length(Rs))
   print("         SHIPBasis: "); @btime SHIPs.eval_basis_d!($b, $db, $B, $Rs, $dbtmp)
   print("         SHIP     : "); @btime SHIPs.evaluate_d!($dEs, $store, $🚢, $Rs)
end


for n = 2:4
end
