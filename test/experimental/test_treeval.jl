
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


##

using SHIPs
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate!, evaluate_d!
using BenchmarkTools, StaticArrays
using Profile

##


basis = SHIPs.Utils.rpi_basis(N = 8, maxdeg = 18)
length(basis)
length(basis.pibasis)
V = SHIPs.Random.randcombine(basis)
Rs, Zs, z0 = rand_nhd(15, basis.pibasis.basis1p.J)
tmp = SHIPs.alloc_temp(V, length(Rs))

# @btime evaluate!($tmp, $V, $Rs, $Zs, $z0);


A = real(tmp.tmp_pibasis.A)

function naiveeval(orders, iAA2iA, coeffs, A)
   V = 0.0
   @inbounds begin
      for iAA = 1:size(iAA2iA, 1)
         v = 1.0
         for a =  1:orders[iAA]
            v *= coeffs[iAA] * A[a]
         end
         V += v
      end
   end
   return V
end


function faketreeeval(inds, coeffs, A, ::Val{M}, AA = zero(MVector{M, Float64})) where {M}
   i1 = length(A)
   V = 0.0
   @inbounds begin
      for i = 1:length(A)
         AA[i] = A[i]
         V += coeffs[i] * AA[i]
      end
      i2 = M
      for i = i1+1:i2
         ip = inds[i]
         aa = AA[ip[1]] * AA[ip[2]]
         AA[i] = aa
         V += coeffs[i] * aa
      end
      i3 = length(coeffs)
      for i = i2+1:i3
         ip = inds[i]
         V += coeffs[i] * AA[ip[1]] * AA[ip[2]]
      end
   end
   return V
end


orders = V.pibasis.inner[1].orders
iAA2iA = V.pibasis.inner[1].iAA2iA
coeffs = V.coeffs[1]
inds = iAA2iA[:,1]
AA = zeros(length(inds))

M = 2000
inds = [ ( Int(rand(1:M)), Int(rand(1:M)) )
            for i = 1:length(coeffs) ]
vM = Val(M)
AAs = zero(MVector{M, Float64})
AAh = zeros(M)

##

print("     Naive Eval:")
@btime naiveeval($orders, $iAA2iA, $coeffs, $A);
print("Tree Eval alloc:")
@btime faketreeeval($inds, $coeffs, $A, $vM);
print("Tree Eval stack:")
@btime faketreeeval($inds, $coeffs, $A, $vM, $AAs);
print(" Tree Eval heap:")
@btime faketreeeval($inds, $coeffs, $A, $vM, $AAh);

# ##
#
#
#
# Profile.clear()
# @profile begin
#    for n = 1:100_000; evaluate!(tmp, V, Rs, Zs, z0); end
# end
# ##
#
# Profile.print()
#
# ##
#
# using ProfileView
# ProfileView.view()
