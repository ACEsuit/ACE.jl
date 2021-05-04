
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------

using JuLIP, ACE, JuLIP.Potentials, LinearAlgebra
using ACE.Testing: lsq, get_V0
using LinearAlgebra: qr, cond
using JuLIP: Atoms
using Plots

function rand_config(V; rattle = 0.2, nrepeat = 3)
   at = bulk(:W, cubic=true, pbc=false) * nrepeat
   return rattle!(at, rattle)
end

function get_2b_basis(species; maxdeg = 10, rcut = 8.0 )
   basis = pair_basis(species = species,
      r0 = rnn(Symbol(species)),
      maxdeg = maxdeg,
      rcut = rcut)
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

#set potential
pot = LennardJones(rnn(:W), 1.0) * C2Shift(6.0)

#create train database
train = trainset(pot, 1000)

#set up 2b basis, and perform the least squares problem
basis = get_2b_basis(:W; maxdeg = 14, rcut = 6.0)
c, relrmse = lsq(train, basis, wE=30.0, wF=1.0)

#combine with coefficients to get IP model
IP = JuLIP.MLIPs.combine(basis, c)

E_pot = []
E_ip = []
R = []

for r in range(1,10,length=1000)
   X = [[0.0,0.0,0.0], [0.0, 0.0, r]]
   C = [[100.0,0.0,0.0], [0.0, 100.0, 0.0],[0.0, 0.0, 100.0] ]
   at = Atoms(X, [[0.0,0.0,0.0], [0.0, 0.0, 0.0]], [0.0, 0.0], AtomicNumber.([74, 74]), C, false)
   push!(R,r)
   push!(E_ip, energy(IP, at))
   push!(E_pot, energy(pot,at))
end

plot(R,E_pot, label="LJ")
plot!(R,E_ip, label="FIT")
ylims!((-1,1))
xlims!((2,8))

#FORCE SCATTER PLOT ON TEST SET
F_ip = []
F_2b = []

for i in 1:10
   at = rand_config(pot)
   push!(F_ip, vcat(forces(IP, at)...))
   push!(F_2b, vcat(forces(pot, at)...))
end

scatter(vcat(F_ip...), vcat(F_2b...))
