
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



module Wigner
   using PyCall, StaticArrays, Cubature

   sympy = pyimport("sympy")
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


   function symprodD(ll, mm, kk)
      a = sympy.Symbol("a")
      b = sympy.Symbol("b")
      c = sympy.Symbol("c")
      p = Rotation.D(ll[1], mm[1], kk[1], a, b, c)
      for i = 2:length(ll)
         p = p * Rotation.D(ll[i], mm[i], kk[i], a, b, c)
      end
      return p
   end

   function symC(ll, mm, kk)
      a = sympy.Symbol("a")
      b = sympy.Symbol("b")
      c = sympy.Symbol("c")
      p = Rotation.D(ll[1], mm[1], kk[1], a, b, c)
      for i = 2:length(ll)
         p = p * Rotation.D(ll[i], mm[i], kk[i], a, b, c)
      end
      spi = sympy.pi
      Ip = sympy.integrate(p, (a, 0, 2*spi), (b, 0, spi), (c, 0, 2*spi))
      Ip.doit().evalf()
   end
end


ll = SVector(2,2,2)
mm = SVector(-1,2,-1)
kk = SVector(0,2,-2)
Wigner.symC(ll, mm, kk)
