
# This script experiments with AD-ing a composition of functions 
# that involves a transform from a property to a number via `val`.
# 
# At some point there was a concern that we need to fully rrule the 
# outer function as well, but this test here shows this is not necessary.
# Even if the rrules return Vectors of invariants (or similar), then 
# Zygote with convert them, using the `ProjectTo`, which is implemented 
# in properties.jl: 
#   (::ProjectTo{T})(φ::Invariant{T}) where {T} = val(φ)
#
# Another key point to note here (not really related to val I think)
# is that using Zygote inside the rrules means that SVectors are 
# converted to SizedVectors; sticking with SVectors requires manual 
# implementation of the rrules. But it is not entirely clear to me why.

##

using ACE, StaticArrays, Zygote, ChainRules, LinearAlgebra
using ACE: Invariant, val 

import ChainRules: rrule, NoTangent, ZeroTangent

##

function inner(x)
   xi1 = Invariant.(x[1]) 
   xi2 = Invariant.(x[2])
   return SVector(xi1[1] * xi1[2], xi2[1] + xi2[2])
end

function rrule_inner(dp, x, T=Invariant) 
   xi1 = T.(x[1]) 
   xi2 = T.(x[2])
   o = T(1.0)
   g = [ SA[ dp[1] * xi1[2], dp[1] * xi2[1] ], 
         SA[ dp[2] * o, dp[2] * o ] ]
   @show eltype(g)          
   return NoTangent(), g 
end

rrule(::typeof(inner), x) = 
         inner(x), dp -> rrule_inner(dp, x)

function rrule2_inner(dq, dp, x)
   @assert dq[1] isa ZeroTangent
   @assert length(dq) == 2
   dq2 = dq[2] 
   @assert dq2 isa AbstractVector{<: SVector}
   # now we need 
   #  ∂(dq[2] ⋅ g) / ∂dq and  ∂(dq[2] ⋅ g) / ∂x

   # here I'm hacking a bit so I can use Zygote to take the second 
   # derivative for me but then convert back to invariants to make sure 
   # the "units" are correct. Note that `g` and hence `dq[2] ⋅ g)` should have 
   # both be invariants!!
   f(dp_, x_) = ACE.contract(dq2, rrule_inner(dp_, x_, identity)[2])
   gg = Zygote.gradient(f, dp, x)
   gg_dp = SVector{2, Invariant{Float64}}(gg[1])
   gg_x = SVector{2, Invariant{Float64}}.(gg[2])
   @show eltype(gg_dp)
   @show eltype(gg_x)
   return NoTangent(), gg_dp, gg_x
end

rrule(::typeof(rrule_inner), dp, x) = 
         rrule_inner(dp, x), dq -> rrule2_inner(dq, dp, x)

##

outer(m) = sum( val.(m).^2 )

toy(x) = outer(inner(x))

##

x = randn(SVector{2, Float64}, 2)
toy(x)
Zygote.gradient(toy, x)[1]

##
toy2(x) = sum(ACE.normsq, Zygote.gradient(toy, x)[1])
toy2(x)

Zygote.refresh()

Zygote.gradient(toy2, x)