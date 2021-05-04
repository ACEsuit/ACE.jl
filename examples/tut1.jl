
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------

using JuLIP, ACE, JuLIP.Potentials, LinearAlgebra
using ACE.Testing: lsq, get_V0
using LinearAlgebra: qr, cond
using Plots

function rand_config(V; rattle = 0.4, nrepeat = 3)
   at = bulk(:W, cubic=true, pbc=false) * nrepeat
   return rattle!(at, rattle)
end

function get_basis(species; N = 3, maxdeg = 10, rcut = 7.0 )
   rcut = 7.0
   basis = ACE.Utils.rpi_basis(; species=species, N = N, r0 = 2.5,
   maxdeg = maxdeg, rcut = rcut,
   rin = rnn(:W) * 0.6,
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

function trainset(Vref, Ntrain; kwargs...)
   train = []
   for n = 1:Ntrain
      at = rand_config(Vref)
      push!(train, (at = at, E = energy(pot, at), F = forces(pot, at)))
   end
   V0 = get_V0(train)
   @show V0
   train = [ (at = at, E = E - energy(V0, at), F = F)
               for (at, E,  F) in train ]
   return train
end

species = :W
pot = EAM("w_eam4.fs")

train = trainset(pot, 100)

basis = get_basis(:W; N = 3, maxdeg = 12, rcut = 6.0)
c, relrmse = lsq(train, basis)

IP = JuLIP.MLIPs.combine(basis, c)

#FORCE SCATTER PLOT ON TEST SET
F_ip = []
F_eam = []

for i in 1:10
   at = rand_config(pot)
   push!(F_ip, vcat(forces(IP, at)...))
   push!(F_eam, vcat(forces(pot, at)...))
end

scatter(vcat(F_ip...), vcat(F_eam...))
