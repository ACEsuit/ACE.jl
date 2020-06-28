
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

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
