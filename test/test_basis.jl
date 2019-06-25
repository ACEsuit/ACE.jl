
@testset "SHIP Basis" begin

##

@info("-------- TEST 🚢  BASIS ---------")
using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using SHIPs: eval_basis!, eval_basis
using JuLIP.MLIPs: IPSuperBasis

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ]
function randiso()
   K = @SMatrix rand(3,3)
   K = K - K'
   Q = rand([-1,1]) * exp(K)
end
function randiso(Rs)
   Q = randiso()
   return [ Q * R for R in shuffle(Rs) ]
end

##

trans3 = PolyTransform(3, 1.0)
ship3 = SHIPBasis(3, 13, 2.0, trans3, 2, 0.5, 3.0)
trans2 = PolyTransform(2, 1.3)
ship2 = SHIPBasis(2, 15, 2.0, trans2, 2, 0.5, 3.0)
ship4 = SHIPBasis(4, 11, 1.0, trans3, 2, 0.5, 3.0)
ships = [ship2, ship3, ship4]

@info("Test (de-)dictionisation of basis sets")
println(@test (decode_dict(Dict(ship2)) == ship2))
println(@test (decode_dict(Dict(ship3)) == ship3))
println(@test (decode_dict(Dict(ship4)) == ship4))
@info("Test (de-)dictionisation of SuperBasis")
super = IPSuperBasis(ship2, ship3, ship4)
println(@test (decode_dict(Dict(super)) == super))

@info("Test isometry invariance for 3B, 4B and 5B 🚢 s")
for ntest = 1:20
   Rs = randR(20)
   BB = [ eval_basis(🚢, Rs) for 🚢 in ships ]
   RsX = randiso(Rs)
   BBX = [ eval_basis(🚢, RsX) for 🚢 in ships ]
   for (B, BX) in zip(BB, BBX)
      print_tf(@test B ≈ BX)
   end
end
println()

##
@info("Test gradients for 3B, 4B and 5B 🚢 s")
for 🚢 in ships
   @info("  body-order = $(SHIPs.bodyorder(🚢)+1):")
   Rs = randR(20)
   store = SHIPs.alloc_temp_d(🚢, Rs)
   SHIPs.precompute_grads!(store, 🚢, Rs)
   B1 = eval_basis(🚢, Rs)
   B = SHIPs.alloc_B(🚢)
   dB = SHIPs.alloc_dB(🚢, Rs)
   SHIPs.eval_basis_d!(B, dB, 🚢, Rs, store)
   @info("      check the basis and basis_d co-incide exactly")
   println(@test B == B1)
   @info("      finite-difference test into random directions")
   for ndirections = 1:20
      Us = randR(length(Rs))
      errs = Float64[]
      for p = 2:10
         h = 0.1^p
         Bh = eval_basis(🚢, Rs+h*Us)
         dBh = (Bh - B) / h
         dBxU = sum( dot.(Ref(Us[n]), dB[n,:])  for n = 1:length(Rs) )
         push!(errs, norm(dBh - dBxU, Inf))
      end
      success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
      print_tf(@test success)
   end
   println()
end


##
verbose=false
@info("Test gradients for 3B with and R near the pole")
🚢 = ship2 = SHIPBasis(2, 15, 2.0, PolyTransform(2, 1.3), 2, 0.5, 3.0)
@info("  body-order = $(SHIPs.bodyorder(🚢)+1):")
# Rs = [ randR(5); [SVector(1e-14*rand(), 1e-14*rand(), 1.1+1e-6*rand())] ]
Rs = [ randR(5); [SVector(0, 0, 1.1+0.5*rand())]; [SVector(1e-14*rand(), 1e-14*rand(), 0.9+0.5*rand())] ]
store = SHIPs.alloc_temp_d(🚢, Rs)
SHIPs.precompute_grads!(store, 🚢, Rs)
B1 = eval_basis(🚢, Rs)
B = SHIPs.alloc_B(🚢)
dB = SHIPs.alloc_dB(🚢, Rs)
SHIPs.eval_basis_d!(B, dB, 🚢, Rs, store)
@info("      finite-difference test into random directions")
for ndirections = 1:30
   Us = randR(length(Rs))
   errs = Float64[]
   for p = 2:10
      h = 0.1^p
      Bh = eval_basis(🚢, Rs+h*Us)
      dBh = (Bh - B) / h
      dBxU = sum( dot.(Ref(Us[n]), dB[n,:])  for n = 1:length(Rs) )
      push!(errs, norm(dBh - dBxU, Inf))
      verbose && (@printf("  %2d | %.2e \n", p, errs[end]))
   end
   success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()


end
