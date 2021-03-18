
module Experimental

include("aceimports.jl")
include("extimports.jl")

struct Pure2BBasis{TB} <: IPBasis
   basis::TB
end

Base.length(basis::Pure2BBasis, args...) = length(basis.basis, args...)
fltype(basis::Pure2BBasis) = fltype(basis.basis)
zlist(basis::Pure2BBasis) = zlist(basis.basis)
cutoff(basis::Pure2BBasis) = cutoff(basis.basis)
_basisfcnidx(basis::Pure2BBasis, iz0::Integer, ib::Integer) =
         _basisfcnidx(basis.basis, iz0, ib)
==(B1::Pure2BBasis, B2::Pure2BBasis) = (B1.pibasis == B2.pibasis)
write_dict(basis::Pure2BBasis) = Dict(
      "__id__" => "ACE_Pure2BRPIBasis",
      "basis" => write_dict(basis.basis) )
read_dict(::Val{:ACE_Pure2BRPIBasis}, D::Dict) =
      Pure2BBasis( read_dict(D["basis"]) )


# TODO
# combine(basis::Pure2BRPIBasis, coeffs) =
#
#    picoeffs = ntuple(iz0 -> (coeffs[basis.Bz0inds[iz0]]' * basis.A2Bmaps[iz0])[:],
#                      numz(basis.pibasis))
#    return PIPotential(basis.pibasis, picoeffs)
# end

scaling(basis::Pure2BBasis, args...; kwargs...)
                  = scaling(basis.basis, args...; kwargs...)


# ------------------------------------------------------------------------
#    Evaluation code
# ------------------------------------------------------------------------

alloc_temp(basis::Pure2BBasis, args...) =
   ( tmpbasis = alloc_temp(basis.basis, args...),
   )

function evaluate!(B, tmp, basis::Pure2BBasis, Rs, Zs, z0)
   # WAAHH!!! THIS IS BAD!!!
   iz0 = z2i(basis, z0)
   AA = site_evaluate!(tmp.AA, tmp.tmp_pibasis, basis.pibasis, Rs, Zs, z0)
   # TODO: this could be done better maybe by adding the real function into
   #       site_evaluate!, or by writing a real version of it...
   AAr = @view tmp.AAr[1:length(AA)]
   AAr[:] .= real.(AA)
   Bview = @view B[basis.Bz0inds[iz0]]
   mul!(Bview, basis.A2Bmaps[iz0], AAr)
   return B
end
