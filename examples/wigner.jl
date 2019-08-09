
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



module Wigner
   using PyCall, StaticArrays, Cubature

   sympy_spin = pyimport("sympy.physics.quantum.spin")
   Rotation = sympy_spin.Rotation

   function D(j, m, mp, α, β, γ)::ComplexF64
      return Rotation.D(j, m, mp, α, β, γ).doit().evalf()
   end

   function prodD(ll, mm, kk, αβγ)
      p = 1.0*im
      for i = 1:length(ll)
         p *= D(ll[i], mm[i], kk[i], αβγ...)::ComplexF64
      end
      return p
   end

   function quad_prodD(ll, mm, kk; kwargs...)
      # ∫dα ∫sinβdβ ∫dγ
      # where α, γ ∈ [0, 2π], β ∈ [0, π]
      f = αβγ -> real( prodD(ll, mm, kk, αβγ) * sin(αβγ[2]) )
      xmin = [0.0, 0.0, 0.0]
      xmax = [2*π, π, 2*π]
      return hcubature(f, xmin, xmax; kwargs...)
   end

end


ll = SVector(2,2,2)
mm = SVector(-1,2,-1)
kk = SVector(0,2,-2)
Wigner.quad_prodD(ll, mm, kk; reltol=1e-2)
