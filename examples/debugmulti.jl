
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using JuLIP, SHIPs, JuLIP.Potentials, LinearAlgebra
using JuLIP.Potentials: SZList, ZList


module ToyM

   using JuLIP, JuLIP.Potentials, LinearAlgebra

   import JuLIP: energy
   import JuLIP.Potentials: SZList, F64fun, ZList, z2i

   mutable struct ToyModel{NZ}
      β::Float64
      eF::Float64
      zlist::SZList{NZ}
      α::Matrix{Float64}
   end

   function ToyModel(species; α = :rand, β = 1.0)
      zlist = ZList(species, static=true)
      if α == :rand
         α = 1.0 .+ 3 * rand(length(zlist), length(zlist))
         α = 0.5 * (α + α')
      end
      V = ToyModel(β, 0.0, ZList(species, static=true), α)
      V.eF = find_eF(V)
      return V
   end

   function find_eF(V)
      H = hamiltonian(rand_config(V), V)
      emin, emax = extrema(eigvals(H))
      for n = 1:30
         emin, emax = extrema([eigvals(H); [emin, emax]])
      end
      return 0.5 * (emin+emax)
   end

   fd(e, β) = 1/(1 + exp(β * e))

   function hamiltonian(at, V)
      X = positions(at)
      Z = at.Z
      H = zeros(length(at), length(at))
      for i = 1:length(at), j = 1:length(at)
         α = V.α[ z2i(V.zlist, Z[i]), z2i(V.zlist, Z[j]) ]
         H[i, j] = exp(- α/3 * norm(X[i]-X[j]))
      end
      return H
   end

   function energy(V::ToyModel, at::Atoms)
      H = hamiltonian(at, V)
      e = eigvals(H)
      # @show extrema(e)
      return sum(fd.(e .- 3, V.β))
   end


   function rand_config(V, rattle = 0.2)
      at = bulk(:Fe, cubic=true, pbc=false) * 3
      for n = 1:length(at)
         at.Z[n] = rand(V.zlist.list)
      end
      return rattle!(at, rattle)
   end
end


#---

function get_V0(train)
   # get list of atomic numbers
   Zs = AtomicNumber[]
   for (at, E) in train
      Zs = unique( [Zs; at.Z] )
   end

   # setup lsq system for E0s
   A = zeros(length(train), length(Zs))
   y = zeros(length(train))
   for (it, (at, E)) in enumerate(train)
      y[it] = E
      for (iz, z) in enumerate(Zs)
         A[it, iz] = length(findall(at.Z .== z))
      end
   end
   E0s = A \ y
   @show E0s
   syms = chemical_symbol.(Zs)
   return JuLIP.OneBody([syms[i] => E0s[i] for i = 1:length(Zs)]...)
end

function trainset(Vref, Ntrain)
   configs = []
   for n = 1:Ntrain
      at = ToyM.rand_config(Vref)
      push!(configs, (at = at, E = energy(Vref, at)))
   end
   # remove E0s
   V0 = get_V0(train)
   configs = [ (at = at, E = E - energy(V0, at)) for (at, E) in configs ]
   return configs
end

function get_basis(species; N = 3, maxdeg = 10, rcut = 7.0 )
   rcut = 7.0
   basis = SHIPs.Utils.rpi_basis(; species=species, N = N, r0 = 2.5,
                                   maxdeg = maxdeg, rcut = rcut,
                                   rin = rnn(:Fe) * 0.6,
                                   constants = false )
   return basis
end

function lsq(train, basis)
   @show length(train)
   @show length(basis)
   A = zeros(length(train), length(basis))
   y = zeros(length(train))
   for (irow, (at, E)) in enumerate(train)
      A[irow, :] =  energy(basis, at)
      y[irow] = E
   end
   F = qr(A)
   @show cond(F.R)
   c = F \ y
   relrmse = norm(A * c - y) / norm(y)
   @show relrmse
   return c
end

#---

species = [:Al, :Fe]
Vref = ToyM.ToyModel(species)
train = trainset(Vref, 20_000)

for maxdeg in [6, 8, 10, 12]
   B = get_basis(species; N = 4, maxdeg = maxdeg, rcut = 7.0)
   lsq(train, B)
end
