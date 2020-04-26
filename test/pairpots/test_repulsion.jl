
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



@testset "RepulsiveCore" begin

##
using Test
using SHIPs, JuLIP, LinearAlgebra, Test
using JuLIP.Testing, JuLIP.MLIPs
randr() = 1.0 + rand()
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)


##
@info("--------------- Testing RepulsiveCore Implementation ---------------")

at = bulk(:W, cubic=true) * 3
rattle!(at, 0.03)
r0 = rnn(:W)
z = atomic_number(:W)
trans = PolyTransform(2, r0)
fcut = PolyCutoff2s(2, 0.5*r0, 1.95*r0)
B = PolyPairBasis(:W, 10, trans, fcut)
coeffs = randcoeffs(B)
pot = combine(B, coeffs)

## try out the repulsive potential
Vfit = pot

ri = 2.1
@show @D Vfit(ri)
if (@D Vfit(ri)) > 0
   Vfit = PolyPairPot(- Vfit.coeffs, Vfit.J, Vfit.zlist, Vfit.bidx0)
end
@show @D Vfit(ri)
e0 = Vfit(ri) - 1.0
Vrep = SHIPs.Repulsion.RepulsiveCore(Vfit, ri)


rout = range(ri+1e-15, 4.0, length=100)
println(@test all(Vfit(r) == Vrep(r,z,z) for r in rout))
rin = range(0.5, ri, length=100)
println(@test all(Vrep.Vin[1](r) == Vrep(r,z,z) for r in rin))

@info("JuLIP FD test")
println(@test JuLIP.Testing.fdtest(Vrep, at))

@info("check scaling")
println(@test energy(Vfit, at) ≈ energy(Vrep, at))

##


@info("--------------- Multi-Species RepulsiveCore ---------------")

at = bulk(:W, cubic=true) * 3
at.Z[2:3:end] .= atomic_number(:Fe)
rattle!(at, 0.03)
r0 = rnn(:W)
trans = PolyTransform(2, r0)
fcut = PolyCutoff2s(2, 0.5*r0, 1.95*r0)
B = PolyPairBasis([:W, :Fe], 10, trans, fcut)
coeffs = randcoeffs(B)
pot = combine(B, coeffs)

## try out the repulsive potential
Vfit = pot

ri = 2.1
z1 = 26
z2 = 74
@show @D Vfit(ri, z1, z2)
e0 = min(Vfit(ri, z1, z2), Vfit(ri, z1, z1), Vfit(ri, z2, z2)) - 1.0
Vrep = SHIPs.Repulsion.RepulsiveCore(Vfit, ri, e0)

for (z, z0) in zip([z1, z1, z2], [z1, z2, z2])
   i, i0 = JuLIP.Potentials.z2i(Vfit, z), JuLIP.Potentials.z2i(Vfit, z0)
   rout = range(ri+1e-15, 4.0, length=100)
   println(@test all(Vfit(r, z, z0) == Vrep(r, z, z0) for r in rout))
   rin = range(0.5, ri, length=100)
   println(@test all(Vrep.Vin[i,i0](r) == Vrep(r, z, z0) for r in rin))
end


@info("JuLIP FD test")
println(@test JuLIP.Testing.fdtest(Vrep, at))

@info("check scaling")
println(@test energy(Vfit, at) ≈ energy(Vrep, at))



@info("check FIO")
println(@test (Vrep == decode_dict(Dict(Vrep))))
fname = tempname()
save_json(fname, Dict(Vrep))
D1 = load_json(fname)
rm(fname)
println(@test (Vrep == decode_dict(D1)))

##

end
