
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using LinearAlgebra: qr, cond

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
   @info("  E0s = $(E0s)")
   syms = chemical_symbol.(Zs)
   return JuLIP.OneBody([syms[i] => E0s[i] for i = 1:length(Zs)]...)
end


function trainset(Vref, Ntrain; kwargs...)
   train = []
   for n = 1:Ntrain
      at = rand(Vref; kwargs...)
      push!(train, (at = at, E = energy(Vref, at), F = forces(Vref, at)))
   end
   # remove E0s
   V0 = get_V0(train)
   train = [ (at = at, E = E - energy(V0, at), F = F)
               for (at, E,  F) in train ]
   return train
end



function lsq(train, basis; verbose=true, wE = 1.0, wF = 1.0)
   @info("lsq info")
   nobs = sum( 1+3*length(t.at) for t in train )
   @info("  nobs = $(nobs); nbasis = $(length(basis))")
   A = zeros(nobs, length(basis))
   y = zeros(nobs)
   irow = 0
   for (at, E, F) in train
      # add energy to the lsq system
      irow += 1
      y[irow] = wE * E / length(at)
      A[irow, :] = wE * energy(basis, at) / length(at)

      # add forces to the lsq system
      nf = 3*length(at)
      y[(irow+1):(irow+nf)] = wF * mat(F)[:]
      Fb = forces(basis, at)
      for ib = 1:length(basis)
         A[(irow+1):(irow+nf), ib] = wF * mat(Fb[ib])[:]
      end
      irow += nf
   end
   qrF = qr(A)
   c = qrF \ y
   relrmse = norm(A * c - y) / norm(y)
   @info("   cond(R) = $(cond(qrF.R)); relrmse = $(relrmse)")
   return c, relrmse
end
