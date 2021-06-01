
import ACE.SphericalHarmonics: SHBasis, index_y


@doc raw"""
`struct Ylm1pBasis <: OneParticleBasis`

One-particle basis of the form
```math
\phi_{lm}({\bm r}) = Y_l^m(\hat{\br r})
```
Fundamental building block of ACE basis sets of the form
```math
   R_{nl}^{\mu_i, \mu_j}(r_{ij}) Y_l^m(\hat{\bm r})
```
This type basically just translates the `SHBasis` into a valid one-particle
basis.
"""
mutable struct Ylm1pBasis{T, VSYM, LSYM, MSYM} <: OneParticleBasis{Complex{T}}
   SH::SHBasis{T}  # SH = Ylm
end

# ---------------------- Implementation of Ylm1pBasis


Ylm1pBasis(maxL::Integer, T = Float64; kwargs...) = 
      Ylm1pBasis((SHBasis(maxL, T)); kwargs...)

Ylm1pBasis(SH::SHBasis{T}; varsym = :rr, lsym = :l, msym = :m)  where {T} = 
   Ylm1pBasis{T, varsym, lsym, msym}(SH)

Base.length(basis::Ylm1pBasis) = length(basis.SH)

_varsym(::Ylm1pBasis{T, VSYM, LSYM, MSYM}) where {T, VSYM, LSYM, MSYM} = VSYM
_lsym(::Ylm1pBasis{T, VSYM, LSYM, MSYM}) where {T, VSYM, LSYM, MSYM} = LSYM
_msym(::Ylm1pBasis{T, VSYM, LSYM, MSYM}) where {T, VSYM, LSYM, MSYM} = MSYM
_l(b, basis::Ylm1pBasis) = getproperty(b, _lsym(basis))
_m(b, basis::Ylm1pBasis) = getproperty(b, _msym(basis))
_lm(b, basis::Ylm1pBasis) = _l(b, basis), _m(b, basis)

# -> TODO : figure out how to do this well!!!
# Base.rand(basis::Ylm1pBasis) =
#       AtomState(rand(basis.zlist.list), ACE.Random.rand_vec(basis.J))

function get_spec(basis::Ylm1pBasis{T, VS, L, M}) where {T, VS, L, M}
   @assert length(basis) == (basis.SH.maxL + 1)^2
   spec = Vector{NamedTuple{(L, M), Tuple{Int, Int}}}(undef, length(basis))
   for l = 0:basis.SH.maxL, m = -l:l
      spec[index_y(l, m)] = NamedTuple{(L, M), Tuple{Int, Int}}( (l, m) )
   end
   return spec
end

==(P1::Ylm1pBasis, P2::Ylm1pBasis) =  ACE._allfieldsequal(P1, P2)

write_dict(basis::Ylm1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_Ylm1pBasis",
          "SH" => write_dict(basis.SH), 
          "varsym" => _varsym(basis),
          "lsym" => _lsym(basis),
          "msym" => _msym(basis) )
   
read_dict(::Val{:ACE_Ylm1pBasis}, D::Dict) = 
      Ylm1pBasis(read_dict(D["SH"]), 
                 varsym = D["varsym"], 
                 lsym = D["lsym"], 
                 msym = D["msym"] )

fltype(basis::Ylm1pBasis{T}) where T = Complex{T}
rfltype(basis::Ylm1pBasis{T}) where T = T

gradtype(basis::Ylm1pBasis{T}, X::TX) where {T, TX <: AbstractState} = 
      promote_type(fltype(basis), TX)

symbols(basis::Ylm1pBasis) = [_lsym(basis), _msym(basis)]


function indexrange(basis::Ylm1pBasis)
   maxl = basis.SH.maxL
   # note we create a stupid tensor product domain and then make up for it
   # by using an index filter during the basis generation process
   return NamedTuple{(_lsym(basis), _msym(basis))}( (0:maxl, -maxl:maxl) )
end


function isadmissible(b, basis::Ylm1pBasis) 
   l, m = _lm(b, basis)
   return (0 <= l <= basis.SH.maxL) && (-l <= m <= l) 
end


# ---------------------------  Evaluation code
#

alloc_B(basis::Ylm1pBasis, args...) = alloc_B(basis.SH)

alloc_dB(basis::Ylm1pBasis, X::AbstractState) = 
      zeros(gradtype(basis, X), length(basis)) 

alloc_temp(basis::Ylm1pBasis, args...) = alloc_temp(basis.SH)

alloc_temp_d(basis::Ylm1pBasis, args...) = 
   ( alloc_temp_d(basis.SH, args...)..., 
     Bsh = alloc_B(basis.SH), 
     dBsh = alloc_dB(basis.SH)
   )


_rr(X, basis::Ylm1pBasis) = getproperty(X, _varsym(basis))

evaluate!(B, tmp, basis::Ylm1pBasis, X::AbstractState) =
      evaluate!(B, tmp, basis.SH, _rr(X, basis))

evaluate_d!(dB, tmpd, basis::Ylm1pBasis, X::AbstractState) = 
      (evaluate_ed!(tmpd.Bsh, dB, tmpd, basis, X); dB)

function evaluate_ed!(B, dB, tmpd, basis::Ylm1pBasis, X::AbstractState)
   TDX = eltype(dB)
   RSYM = _varsym(basis)
   evaluate_ed!(B, tmpd.dBsh, tmpd, basis.SH, _rr(X, basis))
   for n = 1:length(basis)
      dB[n] = TDX( NamedTuple{(RSYM,)}((tmpd.dBsh[n],)) )
   end
   return nothing 
end
   

degree(b, Ylm::Ylm1pBasis) = _l(b, Ylm)

degree(b, Ylm::Ylm1pBasis, weight::Dict) = weight[_lsym(Ylm)] * degree(b, Ylm)

get_index(Ylm::Ylm1pBasis, b) = index_y(_l(b, Ylm), _m(b, Ylm))



#
# function add_into_A_dA!(A, dA, tmpd, basis::RnYlm1pBasis, R, iz::Integer, iz0::Integer)
#    r = norm(R)
#    R̂ = R / r
#    # evaluate the r-basis and the R̂-basis for the current neighbour at R
#    evaluate_d!(tmpd.BJ, tmpd.dBJ, tmpd.tmpdJ, basis.J, r)
#    evaluate_d!(tmpd.BY, tmpd.dBY, tmpd.tmpdY, basis.SH, R)
#    # add the contributions to the A_zklm, ∇A
#    @inbounds for (i, nlm) in enumerate(basis.spec)
#       iY = index_y(nlm.l, nlm.m)
#       A[i] += tmpd.BJ[nlm.n] * tmpd.BY[iY]
#       dA[i] = (tmpd.dBJ[nlm.n] * tmpd.BY[iY]) * R̂ + tmpd.BJ[nlm.n] * tmpd.dBY[iY]
#    end
#    return nothing
# end
