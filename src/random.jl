

module Random

# TODO: rename rand_radial -> rand???

import LinearAlgebra: norm
import ACE: rand_radial, scaling, 
            PositionState, ACEBasis, 
            rand_sphere, rand_rot, rand_refl, rand_O3, 
            B1pComponent

using Random: shuffle

using StaticArrays: @SMatrix, SVector

export rand_nhd, rand_config, rand_sym, randcoeffs, randcombine, rand_vec3

# -------------------------------------------
#   random neighbourhoods and  configurations

rand_radial(Rn::B1pComponent) = 
         Rn.meta["rin"] + rand() * (Rn.meta["rcut"] - Rn.meta["rin"])

rand_vec3(Rn::B1pComponent) = rand_radial(Rn) * rand_sphere()
      

Base.rand(::Type{TX}, basis::ACEBasis) where {TX <: PositionState} =
         TX( (rr = rand_vec3(basis),) )

Base.rand(T::Type{TX}, basis::ACEBasis, N::Integer) where {TX <: PositionState} =
         [ rand(T, basis) for _=1:N ]


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
