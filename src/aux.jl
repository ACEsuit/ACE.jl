
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


"""
Number of polynomials of total degree d with weights w, i.e. the number of
tuples (k1,..., kn) ∈ ℕⁿ where `n = length(w)` such that k ⋅ w ≦ d
"""
_npoly_tot(d, w::Vector) = isempty(w) ? 1 : (
            sum( _npoly_tot(d - w[1]*n, w[2:end]) for n = 0:floor(Int,d/w[1]) )
      )

Main._npoly_tot(10, [1,1,1,1,2,2,2,2]) / factorial(4)



# function generate_tuples(deg, wY::Real, bo)
#    # all possible (k, l) pairs
#    allKL = NamedTuple{(:k, :l, :deg),Tuple{Int64,Int64,Int64}}[]
#    # k + wY * l <= deg
#    for k = 0:deg, l = 0:floor(Int, (deg-k)/wY)
#       push!(allKL, (k=k, l=l, deg=(k+wY*l)))
#    end
#    allKL = allKL[1:6]
#
#    # the first iterm is just (0, ..., 0)
#    # we can choose (k1, l1), (k2, l2) ... by indexing into allKL
#    Nu = ones(Int, 1, bo)  # [1, ..., 1]
#    _deg(ν) = sum( allKL[n].deg for n in ν )
#    # Now we start incrementing until we hit the maximum degree
#    # while retaining the ordering ν₁ ≤ ν₂ ≤ …
#    curidx = bo
#    ν = Nu[end,:]
#    while true
#       # we want to increment `curindex`, but if we've reach the maximum degree
#       # then we need to move to the next index down
#       if _deg(ν) >= deg
#          if curidx == 1
#             # we are done! stop the while loop
#             break
#          end
#          ν[curidx:end] .= ν[curidx-1] + 1
#          curidx -= 1
#       else
#          ν[curidx] += 1
#          Nu = [Nu; ν']
#          curidx = bo
#       end
#    end
#    return allKL, Nu
# end
#
#
# allKL, Nu = generate_tuples(5, 1, 3)
# display(allKL)
# display(Nu)



function generate_tuples(deg, wY::Real, bo)
   # all possible (k, l) pairs
   allKL = NamedTuple{(:k, :l, :deg),Tuple{Int64,Int64,Int64}}[]
   # k + wY * l <= deg
   for k = 0:deg, l = 0:floor(Int, (deg-k)/wY)
      push!(allKL, (k=k, l=l, deg=(k+wY*l)))
   end
   allKL = allKL[1:6]

   # the first iterm is just (0, ..., 0)
   # we can choose (k1, l1), (k2, l2) ... by indexing into allKL
   Nu = ones(Int, 1, bo)  # [1, ..., 1]
   _deg(ν) = sum( allKL[n].deg for n in ν )
   # Now we start incrementing until we hit the maximum degree
   # while retaining the ordering ν₁ ≤ ν₂ ≤ …
   curidx = bo
   ν = Nu[end,:]
   while true
      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down
      if _deg(ν) >= deg
         if curidx == 1
            # we are done! stop the while loop
            break
         end
         ν[curidx:end] .= ν[curidx-1] + 1
         curidx -= 1
      else
         ν[curidx] += 1
         Nu = [Nu; ν']
         curidx = bo
      end
   end
   return allKL, Nu
end


allKL, Nu = generate_tuples(5, 1, 3)
display(allKL)
display(Nu)
