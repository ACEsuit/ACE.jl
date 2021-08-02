
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
mutable struct Ylm1pBasis{T, VSYM, LSYM, MSYM, TDX} <: OneParticleBasis{Complex{T}}
   SH::SHBasis{T}  # SH = Ylm
   B_pool::VectorPool{Complex{T}}
   dB_pool::VectorPool{TDX}
end

# ---------------------- Implementation of Ylm1pBasis


Ylm1pBasis(maxL::Integer, T = Float64; kwargs...) = 
      Ylm1pBasis((SHBasis(maxL, T)); kwargs...)

Ylm1pBasis(SH::SHBasis{T}; varsym = :rr, lsym = :l, msym = :m)  where {T} = 
      Ylm1pBasis{T, varsym, lsym, msym}(SH)

function Ylm1pBasis{T, varsym, lsym, msym}(SH::SHBasis{T}) where {T, varsym, lsym, msym}
   TDX = ACE.DState{(varsym,), Tuple{SVector{3, Complex{T}}}}
   return Ylm1pBasis{T, varsym, lsym, msym, TDX}(
            SH, VectorPool{Complex{T}}(), VectorPool{TDX}() )
end

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
                 varsym = Symbol(D["varsym"]), 
                 lsym = Symbol(D["lsym"]), 
                 msym = Symbol(D["msym"]) )

valtype(basis::Ylm1pBasis{T}, X::AbstractConfiguration) where T = Complex{T}
valtype(basis::Ylm1pBasis{T}, X::AbstractState) where T = Complex{T}

gradtype(basis::Ylm1pBasis, X::AbstractState) = dstate_type(valtype(basis, X), X)

symbols(basis::Ylm1pBasis) = [_lsym(basis), _msym(basis)]

_maxL(Ylm::Ylm1pBasis) = ACE.SphericalHarmonics.maxL(Ylm.SH)

function indexrange(basis::Ylm1pBasis)
   maxl = _maxL(basis)
   # note we create a stupid tensor product domain and then make up for it
   # by using an index filter during the basis generation process
   return NamedTuple{(_lsym(basis), _msym(basis))}( (0:maxl, -maxl:maxl) )
end


function isadmissible(b, basis::Ylm1pBasis) 
   l, m = _lm(b, basis)
   maxL = _maxL(basis)
   return (0 <= l <= maxL) && (-l <= m <= l) 
end


# ---------------------------  Evaluation code
#

_rr(X, basis::Ylm1pBasis) = getproperty(X, _varsym(basis))


evaluate!(B, basis::Ylm1pBasis, X::AbstractState) = 
      evaluate!(B, basis.SH, _rr(X, basis))


# TODO -> do we need to revive this? 
# function evaluate_ed!(B, dB, tmpd, basis::Ylm1pBasis, X::AbstractState)
#    TDX = eltype(dB)
#    RSYM = _varsym(basis)
#    evaluate_ed!(B, tmpd.dBsh, tmpd, basis.SH, _rr(X, basis))
#    for n = 1:length(basis)
#       dB[n] = TDX( NamedTuple{(RSYM,)}((tmpd.dBsh[n],)) )
#    end
#    return nothing 
# end


function evaluate_d!(dB, basis::Ylm1pBasis, X::AbstractState)
   TDX = eltype(dB)  # need not be the same as gradtype!!!
   RSYM = _varsym(basis)
   rr = _rr(X, basis)
   Y = acquire_B!(basis.SH, rr)
   dY = acquire_dB!(basis.SH, rr)
   # spherical harmonics does only ed since values are essentially free
   evaluate_ed!(Y, dY, basis.SH, rr)
   for n = 1:length(basis)
      dB[n] = TDX( NamedTuple{(RSYM,)}((dY[n],)) )
   end
   release_B!(basis.SH, Y)
   release_dB!(basis.SH, dY)
   return dB 
end



degree(b, Ylm::Ylm1pBasis) = _l(b, Ylm)

degree(b, Ylm::Ylm1pBasis, weight::Dict) = weight[_lsym(Ylm)] * degree(b, Ylm)

get_index(Ylm::Ylm1pBasis, b) = index_y(_l(b, Ylm), _m(b, Ylm))




# -------------- AD 

import ChainRules: rrule, NO_FIELDS, @not_implemented

function _rrule_evaluate(basis::Ylm1pBasis, X::AbstractState, 
                         w::AbstractVector{<: Number})
   dY = evaluate_d(basis, X)
   a = sum( real(w) * real.(d.rr) + imag(w) * imag.(d.rr)
            for (w, d) in zip(w, dY)  )
   TDX = dstate_type(a, X)
   return TDX( NamedTuple{(_varsym(basis),)}( (a,) ) )
end

rrule(::typeof(evaluate), basis::Ylm1pBasis, X::AbstractState) = 
      evaluate(basis, X), 
      w -> (NO_FIELDS, NO_FIELDS, _rrule_evaluate(basis, X, w))


rrule(::typeof(evaluate_d), basis::Ylm1pBasis, X::AbstractState) = 
      evaluate_d(basis, X), 
      w -> (NO_FIELDS, NO_FIELDS, 
            @not_implemented("""Ylm config gradients is currently not implemented
                                composition with vectorial features is therefore 
                                not yet supported."""))
