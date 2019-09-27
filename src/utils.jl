
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module Utils

using LinearAlgebra: norm
using JuLIP: JVecF
using SHIPs: TransformedJacobi,  inv_transform

import Base: rand

function rand_sphere()
   R = randn(JVecF)
   return R / norm(R)
end

function rand_radial(J::TransformedJacobi)
   # uniform sample from [tl, tu]
   x = J.tl + rand() * (J.tu - J.tl)
   # transform back
   return inv_transform(J.trans, x)
end

rand_radial(J::TransformedJacobi, N::Integer) = [ rand_radial(J) for _=1:N ]

rand(J::TransformedJacobi) = rand_radial(J) *  rand_sphere()

rand(J::TransformedJacobi, N::Integer) =  [ rand(J) for _ = 1:N ]


end
