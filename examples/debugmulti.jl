
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



using JuLIP, SHIPs, JuLIP.Potentials, LinearAlgebra
using JuLIP.Potentials: SZList, ZList
using SHIPs.Testing: ToyModel, trainset, lsq

function get_basis(species; N = 3, maxdeg = 10, rcut = 7.0 )
   rcut = 7.0
   basis = SHIPs.Utils.rpi_basis(; species=species, N = N, r0 = 2.5,
   maxdeg = maxdeg, rcut = rcut,
   rin = rnn(:Fe) * 0.6,
   constants = false )
   return basis
end

function calc_rmse(train, Vfit)
   sqe = 0.0
   sqy = 0.0
   for (at, E, F) in train
      sqy += (E/length(at))^2 + norm(mat(F)[:])^2
      Efit = energy(Vfit, at)
      Ffit = forces(Vfit, at)
      sqe += ((E - Efit)/length(at))^2 +
             norm( mat(F)[:] - mat(Ffit)[:] )^2
   end
   return sqrt(sqe / sqy)
end


#---

@info("Quick run + consistency test")
# species = [:Al, :Fe]
species = [:Fe, :Al]
Vref = ToyModel(species)
train = trainset(Vref, 10)
basis = get_basis(species; N = 3, maxdeg = 6, rcut = 6.0)
c, relrmse = lsq(train, basis)

Vfit = JuLIP.MLIPs.combine(basis, c)

norm(forces(Vfit, at) - sum(c .* forces(basis, at)))

relrmse2 = calc_rmse(train, Vfit)
@assert abs(relrmse - relrmse2) < 0.01

#---

species = [:Fe, :Al]
basis = get_basis(species; N = 3, maxdeg = 6, rcut = 6.0)
c = rand(length(basis)) .- 0.5
Vfit = JuLIP.MLIPs.combine(basis, c)
# at = bulk(:Fe, pbc=false, cubic=true) * 3
at = train[1].at
energy(Vfit, at) ≈ sum(c .* energy(basis, at))
forces(Vfit, at) ≈ sum(c .* forces(basis, at))
virial(Vfit, at) ≈ sum(c .* virial(basis, at))


#---
Rs, Zs, z0 = SHIPs.Random.rand_nhd(12, basis.pibasis.basis1p.J, species)
dVfit = SHIPs.evaluate_d(Vfit, Rs, Zs, z0)
dVbas = sum(c_ * dVb
       for (c_, dVb) in zip(c, eachrow(SHIPs.evaluate_d(basis, Rs, Zs, z0)))
   )
dVfit ≈ dVbas

nlist = neighbourlist(at, cutoff(Vfit))

Fbas = zeros(JVecF, length(at))
Ffit = zeros(JVecF, length(at))
for i = 1:length(at)
   global Fbas, dVbas
   j, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, i)
   dVbas = sum(c_ * dVb
       for (c_, dVb) in zip(c, eachrow(SHIPs.evaluate_d(basis, Rs, Zs, at.Z[i])))
          )
   dVfit = SHIPs.evaluate_d(Vfit, Rs, Zs, at.Z[i])
   @assert dVbas ≈ dVfit
   for a = 1:length(Rs)
      Fbas[j[a]] -= dVbas[a]
      Fbas[i]    += dVbas[a]
      Ffit[j[a]] -= dVfit[a]
      Ffit[i]    += dVfit[a]
   end
end

forces(Vfit, at) ≈ Fbas
forces(basis, at)

#--- simple convergence test

species = [:Al, :Fe]
Vref = ToyModel(species)
train = trainset(Vref, 300; rattle=0.1)

for maxdeg in [6, 8, 10, 12]
   B = get_basis(species; N = 6, maxdeg = maxdeg, rcut = 7.0)
   lsq(train, B)
end
