
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using JuLIP

# TODO: Idea => replace (deg, wY) by a "Degree" type and dispatch a lot of
#       functionality on that => e.g. we can then try hyperbolic cross, etc.

using SHIPs.SphericalHarmonics: SHBasis, sizeY, SVec3

# -------------------------------------------------------------
#       construct l,k tuples that specify basis functions
# -------------------------------------------------------------

function generate_LK_tuples(deg, wY::Real, bo)
   # all possible (k, l) pairs
   allKL = NamedTuple{(:k, :l, :deg),Tuple{Int,Int,Float64}}[]
   degs = Int[]
   # k + wY * l <= deg
   for k = 0:deg, l = 0:floor(Int, (deg-k)/wY)
      push!(allKL, (k=k, l=l, deg=(k+wY*l)))
      push!(degs, (k+wY*l))
   end
   # sort allKL according to total degree
   I = sortperm(degs)
   allKL = allKL[I]
   degs = degs[I]

   # the first iterm is just (0, ..., 0)
   # we can choose (k1, l1), (k2, l2) ... by indexing into allKL
   Nu = []
   _deg(ν) = maximum(ν) <= length(allKL) ? sum( allKL[n].deg for n in ν ) : Inf
   # Now we start incrementing until we hit the maximum degree
   # while retaining the ordering ν₁ ≤ ν₂ ≤ …
   lastidx = 0
   ν = MVector(ones(Int, bo)...)
   ctr = 1000
   while true
      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down

      # if the current tuple ν has admissible degree ...
      if _deg(ν) <= deg
         # ... then we add it to the stack  ...
         push!(Nu, SVector(ν))
         # ... and increment it
         lastidx = bo
         ν[lastidx] += 1
      else
         # we have overshot, _deg(ν) > deg; we must go back down, by
         # decreasing the index at which we increment
         if lastidx == 1
            break
         end
         ν[lastidx-1:end] .= ν[lastidx-1] + 1
         lastidx -= 1
      end
   end
   return allKL, Nu
end


# -------------------------------------------------------------
#       define the basis itself
# -------------------------------------------------------------


struct SHIPBasis{BO, T, TJ, TSH}
   deg::Int
   wY::T
   J::TJ
   SH::TSH
   A::Vector{Complex{T}}
   dA::Vector{SVec3{Complex{T}}}
   allKL::NamedTuple{(:k, :l, :deg),Tuple{Int,Int,T}}
   Nu::Vector{SVector{BO, Int}}
   valBO::Val{BO}
end

length_A(deg, wY) = sum( sizeY( floor(Int, (deg - k)/wY) ) for k = 0:deg )

# this could become and allox_temp
alloc_A(deg, wY) = zeros(ComplexF64, length_A(deg, wY))
alloc_dA(deg, wY) = zeros(SVec3{ComplexF64}, length_A(deg, wY))



function SHIPBasis(bo::Integer, deg::Integer, wY::Real, trans, p, rl, ru)
   # r - basis
   maxP = deg
   J = rbasis(maxP, trans, p, rl, ru)
   # R̂ - basis
   maxL = floor(Int, deg / wY)
   SH = SHBasis(maxL)
   # allocate space for the A array
   A = alloc_A(deg, wY)
   dA = alloc_B(deg, wY)
   # get the basis specification
   allKL, Nu = generate_LK_tuples(deg, wY, bo)
   # precompute the Clebsch-Gordan coefficients
   # TODO: maybe later ...
   # putting it all together ...
   return SHIPBasis(deg, wY, J, SH, A, dA, allKL, Nu, Val(bo))
end


bodyorder(ship::SHIPBasis{BO}) where {BO} = BO

length_B(ship::SHIPBasis{BO}) where {BO} = length(ship.Nu)

alloc_B(ship::SHIPBasis) = zeros(Float64, length_B(ship))
alloc_dB(ship::SHIPBasis) = zeros(SVec3{Float64}, length_B(ship))



# -------------------------------------------------------------
#       precompute the J, SH and A arrays
# -------------------------------------------------------------
