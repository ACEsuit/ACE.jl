
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module Exp

using LinearAlgebra
import ACE
using ACE: alloc_B, alloc_temp, evaluate!, order

_get_samples(basis, nargs, nsamples) =
   [  [ ACE.rand_vec(basis.J) for _=1:nargs ]
       for _=1:nsamples ]


function _asmlsq(basis, samples)
   A = zeros(ComplexF64, (length(samples), length(basis)))
   B = alloc_B(basis)
   tmp = alloc_temp(basis)
   Z = zeros(Int, length(samples[1]))
   for (irow, R) in enumerate(samples)
      evaluate!(B, tmp, basis, R, Z, 0)
      A[irow, :] .= B
   end
   return A
end

function determine_order(f, nargs, basis;
                         ntrain = 10_000,
                         ntest = 1_000)
   train = _get_samples(basis, nargs, ntrain)
   test = _get_samples(basis, nargs, ntest)

   # assemble the lsq system
   Y = f.(train)
   A = _asmlsq(basis, train)

   # now loop over all possible body-orders...
   rmse = zeros(order(basis))
   for N = 1:order(basis)
      # indices in the basis corresponding to order N
      IN = ACE.Utils.findall_basis_N(basis, N; verbose=false)
      # the relevant subset of the lsq system is
      AN = A[:, IN]
      # now solve the lsq system and record the relative RMSE
      θ = qr(AN) \ Y
      rmse[N] = norm(AN * θ - Y) / norm(Y)
   end
   return rmse
end


end
