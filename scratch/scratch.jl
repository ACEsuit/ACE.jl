#---


using ACE
using Random, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate_ed
using JuLIP.MLIPs: combine


#---

# for species in (:X, :Si, [:C, :O, :H]), N = 1:length(degrees)

basis = ACE.Utils.rpi_basis(; species = [:Si, :C], N = 3, maxdeg = 6)

#---


Nweight = Dict( 1 => 1.0, 2 => 1.0, 3 => 0.01 )
# ACE.scaling(b::ACE.PIBasisFcn, p) = Nweight[ACE.order(b)] * sum(ACE.scaling(bb, p) for bb in b.oneps)
ACE.scaling(b::ACE.PIBasisFcn, p) = sum(ACE.scaling(bb, p) for bb in b.oneps)

vnew = ACE.scaling(basis, 2)

#---

Nweight = Dict( 1 => 1.0, 2 => 1.0, 3 => 0.01 )
W = [ Nweight[N] * w
      for (N, w) in zip(get_orders(basis), ACE.scaling(basis, 2)) ]

#---

sum(abs.(basis.A2Bmaps[1]), dims=2) |> display


W = [ dot(basis.A2Bmaps[1][i,:], basis.A2Bmaps[1][j,:])
   for i = 1:length(basis), j = 1:length(basis) ]

isdiag(W)

basis.Bz0inds[1]
length(basis.pibasis, 1)
ACE.get_basis_spec(basis.pibasis, 1, 1)
findfirst(basis.A2Bmaps[1][1,:] .!= 0)

o = ACE.RPI.get_orders(basis)

basis.pibasis.inner[1]

@show o

#---

Ns = get_
