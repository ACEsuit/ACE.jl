
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
   label::String
end

# # -------- temporary hack for 1.6, should not be needed from 1.7 onwards 

# function acquire_B!(basis::Ylm1pBasis, args...) 
#    VT = valtype(basis, args...)
#    return acquire!(basis.B_pool, length(basis), VT)
# end

# function release_B!(basis::Ylm1pBasis, B)
#    return release!(basis.B_pool, B)
# end


# ---------------------- Implementation of Ylm1pBasis


Ylm1pBasis(maxL::Integer, T = Float64; kwargs...) = 
      Ylm1pBasis((SHBasis(maxL, T)); kwargs...)

Ylm1pBasis(SH::SHBasis{T}; label = "Ylm", varsym = :rr, lsym = :l, msym = :m)  where {T} = 
      Ylm1pBasis{T, varsym, lsym, msym}(SH, label)

function Ylm1pBasis{T, varsym, lsym, msym}(SH::SHBasis{T}, label) where {T, varsym, lsym, msym}
   TDX = ACE.DState{NamedTuple{(varsym,), Tuple{SVector{3, Complex{T}}}}}
   return Ylm1pBasis{T, varsym, lsym, msym, TDX}(
            SH, VectorPool{Complex{T}}(), VectorPool{TDX}(), 
            label )
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

function get_spec(basis::Ylm1pBasis, i::Integer) 
   l, m = ACE.SphericalHarmonics.idx2lm(i)
   L, M = _lsym(basis), _msym(basis)
   return NamedTuple{(L, M)}((l, m))
end

get_spec(basis::Ylm1pBasis) = get_spec.(Ref(basis), 1:length(basis))

function Base.show(io::IO, basis::Ylm1pBasis)
   print(io, "Ylm1pBasis{$(_varsym(basis)), $(_lsym(basis)), $(_msym(basis))}($(basis.SH.alp.L), \"$(basis.label)\")")
end



# function get_spec(basis::Ylm1pBasis{T, VS, L, M}) where {T, VS, L, M}
#    @assert length(basis) == (_maxL(basis) + 1)^2
#    spec = Vector{NamedTuple{(L, M), Tuple{Int, Int}}}(undef, length(basis))
#    for l = 0:_maxL(basis), m = -l:l
#       spec[index_y(l, m)] = NamedTuple{(L, M), Tuple{Int, Int}}( (l, m) )
#    end
#    return spec
# end

==(P1::Ylm1pBasis, P2::Ylm1pBasis) =  
      ( (P1.SH == P2.SH) && (typeof(P1) == typeof(P2)) )

write_dict(basis::Ylm1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_Ylm1pBasis",
          "SH" => write_dict(basis.SH), 
          "varsym" => _varsym(basis),
          "lsym" => _lsym(basis),
          "msym" => _msym(basis), 
          "label" => basis.label )
   
read_dict(::Val{:ACE_Ylm1pBasis}, D::Dict) = 
      Ylm1pBasis(read_dict(D["SH"]); 
                 varsym = Symbol(D["varsym"]), 
                 lsym = Symbol(D["lsym"]), 
                 msym = Symbol(D["msym"]),
                 label = D["label"] )

# TODO: fix the type promotion...

valtype(basis::Ylm1pBasis{T}, args...) where T = Complex{T}

gradtype(basis::Ylm1pBasis, X::AbstractState) = dstate_type(valtype(basis, X), X)

symbols(basis::Ylm1pBasis) = [_lsym(basis), _msym(basis)]

argsyms(basis::Ylm1pBasis) = (_varsym(basis), )

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


function evaluate_d!(dB, basis::Ylm1pBasis, X::AbstractState)
   B = acquire_B!(basis, X)
   evaluate_ed!(B, dB, basis, X)
   release_B!(basis, B)
   return dB 
end 


function evaluate_ed!(B, dB, basis::Ylm1pBasis, X::AbstractState)
   TDX = eltype(dB)  # need not be the same as gradtype!!!
   RSYM = _varsym(basis)
   rr = _rr(X, basis)
   dY = acquire_dB!(basis.SH, rr)
   # spherical harmonics does only ed since values are essentially free
   evaluate_ed!(B, dY, basis.SH, rr)
   for n = 1:length(basis)
      dB[n] = TDX( NamedTuple{(RSYM,)}((dY[n],)) )
   end
   release_dB!(basis.SH, dY)
   return B, dB 
end



degree(b, Ylm::Ylm1pBasis) = _l(b, Ylm)

degree(b, Ylm::Ylm1pBasis, weight::Dict) = haskey(weight,_lsym(Ylm)) ? weight[_lsym(Ylm)] * degree(b, Ylm) : degree(b, Ylm)

get_index(Ylm::Ylm1pBasis, b) = index_y(_l(b, Ylm), _m(b, Ylm))




# -------------- AD 

import ChainRules: rrule, NoTangent, @not_implemented

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
      w -> (NoTangent(), NoTangent(), _rrule_evaluate(basis, X, w))


rrule(::typeof(evaluate_d), basis::Ylm1pBasis, X::AbstractState) = 
      evaluate_d(basis, X), 
      w -> (NoTangent(), NoTangent(), 
            @not_implemented("""Ylm config gradients is currently not implemented
                                composition with vectorial features is therefore 
                                not yet supported."""))
