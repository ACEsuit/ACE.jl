

module Random

# TODO: rename rand_radial -> rand???

import LinearAlgebra: norm
import ACE: rand_radial, scaling, fltype, rfltype,
            EuclideanVectorState, ACEBasis

using Random: shuffle

using StaticArrays: @SMatrix, SVector

export rand_nhd, rand_config, rand_sym, randcoeffs, randcombine

# -------------------------------------------
#   random neighbourhoods and  configurations

function rand_sphere(T = Float64)
   R = randn(SVector{3, T})
   return R / norm(R)
end

Base.rand(::Type{EuclideanVectorState}, basis::ACEBasis) =
         EuclideanVectorState(rand_radial(basis) * rand_sphere()
      )

Base.rand(T::Type{EuclideanVectorState}, basis::ACEBasis, N::Integer) =
         [ rand(T, basis) for _=1:N ]


rand_sym(Rs, Zs) = rand_refl(rand_rot(rand_perm(Rs, Zs)...)...)

rand_rot() = (K = (@SMatrix rand(3,3)) .- 0.5; exp(K - K'))

rand_refl() = rand([-1,1])


function rand_rot(Xs::AbstractVector)
   Q = rand_rot()
   return [ Q * X for X in Xs ]
end

function rand_refl(Xs::AbstractVector)
   σ = rand_refl()
   return [ σ * X for X in Xs ]
end


# --------------------------------------------------------------
# random operations on neighbourhoods, mostly for testing

# TODO: rewrite for generic ACE

# function rand_perm(Rs, Zs)
#    @assert length(Rs) == length(Zs)
#    p = shuffle(1:length(Rs))
#    return Rs[p], Zs[p]
# end
#
#     random potentials / random basis / random potentials ???

# # TODO: we have an issue with eltypes that needs to be fixed!!!
#
# function randcoeffs(basis; diff = 2)
#    ww = scaling(basis, diff)
#    c = 2 * (rand(rfltype(basis), length(basis)) .- 0.5) ./ ww
#    return c / norm(c)
# end
#
# randcombine(basis; diff = 2) =
#    combine(basis, randcoeffs(basis; diff = diff))

# # move to utility???
# function rand(::Type{ACE.RPI.RPIBasis}; kwargs...)
#
# end

end
