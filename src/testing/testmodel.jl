
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



"""
`mutable struct ToyModel` : a tight-binding type toy model admitting
multiple species, for testing the completeness of basis sets.
"""
mutable struct ToyModel{NZ} <: AbstractCalculator
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
dfd(e, β) = (- β * exp(β * e)) / (1 + exp(β * e))^2

hop(r, α) = exp(-α * (r/3-1))
dhop(r, α) = - α/3 * exp(-α * (r/3-1))

function hamiltonian(at, V)
   X = positions(at)
   Z = at.Z
   H = zeros(length(at), length(at))
   for i = 1:length(at)-1, j = (i+1):length(at)
      α = V.α[ z2i(V.zlist, Z[i]), z2i(V.zlist, Z[j]) ]
      H[j, i] = H[i, j] = hop(norm(X[i]-X[j]), α)
   end
   return H
end

function hamiltonian_d(at::Atoms{T}, V) where {T}
   X = positions(at)
   Z = at.Z
   dH = zeros(JVec{T}, length(at), length(at))
   for i = 1:length(at), j = (i+1):length(at)
      α = V.α[ z2i(V.zlist, Z[i]), z2i(V.zlist, Z[j]) ]
      Rij = X[i]-X[j]
      rij = norm(Rij)
      dH[i, j] = dhop(rij, α) * (Rij / rij)
      dH[j, i] = - dH[i, j]
   end
   return dH
end

function energy(V::ToyModel, at::Atoms)
   H = hamiltonian(at, V)
   e = eigvals(H)
   # @show extrema(e)
   return sum(fd.(e .- V.eF, V.β))
end

function forces(V::ToyModel, at::Atoms)
   H = hamiltonian(at, V)
   e, Ψ = eigen(H)
   dH = hamiltonian_d(at, V)
   F = 2 * [ sum( dfd(e[n] - V.eF, V.β) * sum(Ψ[:, n] .* dH[:, j]) * Ψ[j, n]
                for n = 1:length(e) )
           for j = 1:length(at) ]
   # df_H = sum( dfd(e[n] - V.eF, V.β) * Ψ[:,n] * Ψ[:,n]'  for n = 1:length(e) )
   return F #  - sum(df_H .* dH; dims = 1)[:]
end



Base.rand(V::ToyModel; kwargs...) = rand_config(V; kwargs...)
