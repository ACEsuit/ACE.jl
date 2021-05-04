
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



using JuLIP, ACE, JuLIP.Potentials, LinearAlgebra
using JuLIP.Potentials: SZList, ZList
using ACE.Testing: ToyModel, trainset, lsq

function get_basis(species; N = 3, maxdeg = 10, rcut = 7.0 )
   rcut = 7.0
   basis = ACE.Utils.rpi_basis(; species=species, N = N, r0 = 2.5,
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

species = [:Fe, :Al]
at = ACE.Testing.rand_config(ToyModel(species))

basis = get_basis(species; N = 3, maxdeg = 6, rcut = 6.0)
c = rand(length(basis)) .- 0.5
Vfit = JuLIP.MLIPs.combine(basis, c)
@show energy(Vfit, at) ≈ sum(c .* energy(basis, at))
@show forces(Vfit, at) ≈ sum(c .* forces(basis, at))
@show virial(Vfit, at) ≈ sum(c .* virial(basis, at))


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



#--- simple convergence test

species = [:Al, :Fe]
Vref = ToyModel(species)
train = trainset(Vref, 300; rattle=0.1)

for maxdeg in [6, 8, 10, 12]
   B = get_basis(species; N = 6, maxdeg = maxdeg, rcut = 7.0)
   lsq(train, B)
end
