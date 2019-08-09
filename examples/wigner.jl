
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using StaticArrays, LinearAlgebra

module Wigner
   using PyCall, StaticArrays, Cubature, SHIPs, JuLIP, LinearAlgebra
   using ProgressMeter
   using SHIPs: _mrange

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
      Ip = sympy.integrate(p * sympy.sin(b), (a, 0, 2*spi), (b, 0, spi), (c, 0, 2*spi))
      Ip.doit().evalf()
   end

   function compute_Cmk(ll)
      len = 0
      for mm in SHIPs._mrange(ll)
         len += 1
      end
      Cmk = zeros(ComplexF64, len, len)
      pmtr = Progress(len^2)
      ctr = 0
      for (im, mm) in enumerate(_mrange(ll)), (ik, kk) in enumerate(_mrange(ll))
         Cmk[im, ik] = symC(ll, mm, kk)
         ctr += 1
         update!(pmtr, ctr)
      end
      println()

      return Cmk
   end

   function compute_all_Cmk(maxlen)
      D = Dict("re" => Dict(), "im" => Dict())
      for len = 2:maxlen
         for ill in CartesianIndices( ntuple(_->0:2, len) )
            ll = SVector(Tuple(ill)...)
            @info("Computing the coefficients for $ll")
            C_mk = compute_Cmk(ll)
            @show rank(C_mk)
            D["re"][string(ll)] = real.(C_mk)
            D["im"][string(ll)] = imag.(C_mk)
            JuLIP.save_json("all_Cmk.json", D)
         end
      end
      return D
   end
end


# ll = SVector(2,2,2)
# mm = SVector(-1,2,-1)
# kk = SVector(0,2,-2)
# Wigner.symC(ll, mm, kk)
# ll = SVector(2,2,2)

# ll2a = SVector(2, 3) => Cmk = 0 => checked

# 2b => should get rank-1
#       alternating signs?
# ll2 = SVector(2, 2)
# Cmk = Wigner.compute_Cmk(ll2)
# rank(Cmk)
# svdvals(Cmk)

# 3
# ll3 = SVector(1, 2, 2)
# Cmk = Wigner.compute_Cmk(ll3)
# rank(Cmk)
# svdvals(Cmk)


Wigner.compute_all_Cmk(5)
