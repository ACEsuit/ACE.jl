
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using JuLIP

# TODO: Idea => replace (deg, wY) by a "Degree" type and dispatch a lot of
#       functionality on that => e.g. we can then try hyperbolic cross, etc.

using SHIPs.SphericalHarmonics: SHBasis, sizeY, SVec3

struct SHIPBasis{BO, TJ, TSH}
   deg::Int
   wY::Float64
   J::TJ
   SH::TSH
   A::Vector{Float64}
   valBO::Val{BO}
end

length_A(deg, wY) = sum( sizeY( floor(Int, (deg - k)/wY) ) for k = 0:deg )

# this could become and allox_temp
alloc_A(deg, wY) = zeros(ComplexF64, length_A(deg, wY))
alloc_dA(deg, wY) = zeros(SVec3{ComplexF64}, length_A(deg, wY))


length_B(bo, deg, wY) = Main._npoly_tot(deg, [ ones(bo-1); wY * ones(bo-1) ])
length_B(ship::SHIPBasis{BO}) where {BO} = length_B(BO, ship.deg, ship.wY)

alloc_B(ship::SHIPBasis) = zeros(Float64, length_B(ship))
alloc_dB(ship::SHIPBasis) = zeros(SVec3{Float64}, length_B(ship))



function SHIPBasis(bo::Integer, deg::Integer, wY::Real, trans, p, rl, ru)
   # r - basis
   maxP = deg
   J = rbasis(maxP, trans, p, rl, ru)
   # R̂ - basis
   maxL = floor(Int, deg / wY)
   SH = SHBasis(maxL)
   # allocate space for the A array
   A = Vector{Float64}(undef, length_A(deg, wY))

   # precompute the Clebsch-Gordan coefficients
end




bodyorder(ship::SHIPBasis{BO}) where {BO} = BO
