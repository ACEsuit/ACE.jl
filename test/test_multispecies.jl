
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

##


@info("-------- TEST 🚢  Multi-Species-Basis ---------")
using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using SHIPs: eval_basis!, eval_basis, PolyCutoff1s, PolyCutoff2s
using JuLIP.MLIPs: IPSuperBasis
using JuLIP.Testing: print_tf

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end

randR(N) = [ randR() for n=1:N ], zeros(Int16, N)

randR(N, syms::NTuple{M, Symbol}) where {M} =
      randR(N, Int16.(atomic_number.(syms)))

randR(N, Zs::NTuple{M, <:Integer}) where {M} =
      randR(N, Int16.(Zs))

randR(N, Zs::NTuple{M, Int16}) where {M}  =
      randR(N)[1], rand(Zs, N), rand(Zs)

function randiso()
   K = @SMatrix rand(3,3)
   K = K - K'
   Q = rand([-1,1]) * exp(K)
end

function randiso(Rs, Zs)
   Q = randiso()
   p = shuffle(1:length(Rs))
   return [ Q * R for  R in Rs[p]], Zs[p]
end

##
@info("Testing creation and (de-)dictionisation of a few BasisSpecs")

spec = SparseSHIPBasis(3, :X, 10, 1.5)
println(@test spec == SparseSHIPBasis(3, 10, 1.5))
println(@test decode_dict(Dict(spec)) == spec)

spec = SparseSHIPBasis(3, [:H, :He], 6, 1.5)
println(@test spec == SparseSHIPBasis(3, [1,2], 6, 1.5))
println(@test decode_dict(Dict(spec)) == spec)

spec = SparseSHIPBasis(2, [:H, :He, :Li], 6, 1.5)
println(@test spec == SparseSHIPBasis(2, [1,2,3], 6, 1.5))
println(@test decode_dict(Dict(spec)) == spec)

##

trans = PolyTransform(2, 1.0)
cutf = PolyCutoff2s(2, 0.5, 3.0)

ship2 = SHIPBasis(SparseSHIPBasis(2, [1,2], 10, 2.0), trans, cutf)
ship3 = SHIPBasis(SparseSHIPBasis(3, [1,2], 8, 2.0), trans, cutf)
ship4 = SHIPBasis(SparseSHIPBasis(4, [1,2], 6, 1.5), trans, cutf)
ship5 = SHIPBasis(SparseSHIPBasis(5, [1,2],  5, 1.5), trans, cutf)
ships = [ship2, ship3, ship4, ship5]

@info("Test (de-)dictionisation of basis sets")
for ship in ships
   println(@test (decode_dict(Dict(ship)) == ship))
end


@info("Test isometry invariance for 3B-6B 🚢 s")
for ntest = 1:20
   Rs, Zs, iz = randR(20, (1,2))
   BB = [ eval_basis(🚢, Rs, Zs, iz) for 🚢 in ships ]
   RsX, ZsX = randiso(Rs, Zs)
   BBX = [ eval_basis(🚢, RsX, ZsX, iz) for 🚢 in ships ]
   for (B, BX) in zip(BB, BBX)
      print_tf(@test B ≈ BX)
   end
end
println()


##

@info("Test gradients for 3-6B 🚢-basis")
for 🚢 in ships
   @info("  body-order = $(SHIPs.bodyorder(🚢)):")
   Rs, Zs, z = randR(20, (1,2))
   tmp = SHIPs.alloc_temp_d(🚢, Rs)
   SHIPs.precompute_grads!(tmp, 🚢, Rs, Zs)
   B1 = eval_basis(🚢, Rs, Zs, z)
   B = SHIPs.alloc_B(🚢)
   dB = SHIPs.alloc_dB(🚢, Rs)
   SHIPs.eval_basis_d!(B, dB, tmp, 🚢, Rs, Zs, z)
   @info("      check the basis and basis_d co-incide exactly")
   println(@test B ≈ B1)
   @info("      finite-difference test into random directions")
   for ndirections = 1:20
      Us, _ = randR(length(Rs))
      errs = Float64[]
      for p = 2:10
         h = 0.1^p
         Bh = eval_basis(🚢, Rs+h*Us, Zs, z)
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

@info("Check Correctness of Multi-species SHIPBasis calculators")

randcoeffs(B) = 2 * (rand(length(B)) .- 0.5) .* (1:length(B)).^(-2)

m_naive_energy(basis::SHIPBasis, at) =
      sum( eval_basis(basis, R, at.Z[j], at.Z[i])
            for (i, j, R) in sites(at, cutoff(basis)) )

for basis in ships
   @info("   body-order = $(SHIPs.bodyorder(basis))")
   at = bulk(:Si) * 3
   at.Z[1:2:end] .= 1
   at.Z[2:2:end] .= 2
   rattle!(at, 0.1)
   print("     energy: ")
   println(@test energy(basis, at) ≈ m_naive_energy(basis, at) )
   print("site-energy: ")
   println(@test energy(basis, at) ≈ sum( site_energy(basis, at, n)
                                         for n = 1:length(at) ) )
   # we can test consistency of forces, site energy etc by taking
   # random inner products with coefficients
   # TODO [tuple] revive this test after porting `fast`
   # @info("     a few random combinations")
   # for n = 1:10
   #    c = randcoeffs(basis)
   #    sh = JuLIP.MLIPs.combine(basis, c)
   #    print_tf(@test energy(sh, at) ≈ dot(c, energy(basis, at)))
   #    print_tf(@test forces(sh, at) ≈ sum(c*f for (c, f) in zip(c, forces(basis, at))) )
   #    print_tf(@test site_energy(sh, at, 5) ≈ dot(c, site_energy(basis, at, 5)))
   #    print_tf(@test site_energy_d(sh, at, 5) ≈ sum(c*f for (c, f) in zip(c, site_energy_d(basis, at, 5))) )
   # end
   println()
end
