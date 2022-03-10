

import ACE: OneParticleBasis, AbstractState
import ACE.OrthPolys: TransformedPolys

import NamedTupleTools
using NamedTupleTools: namedtuple


# ------------------------------------------------------------------------

@doc raw"""
`struct XScal1pBasis <: OneParticleBasis`

One-particle basis of the form $P_n(x_i)$ for a general scalar, invariant 
input `x`. This type basically just translates the `TransformedPolys` into a valid
one-particle basis.
"""
mutable struct XScal1pBasis{ISYMS, FVAL, T, TT, TJ, NI, TRG} <: OneParticleBasis{T}
   P::TransformedPolys{T, TT, TJ}
   fval::FVAL    
   rgs::TRG
   spec::Vector{NamedTuple{ISYMS, NTuple{NI, Int}}}
   coeffs::Matrix{T}
   label::String 
end


xscal1pbasis(P::TransformedPolys, idxsyms, valsym::Symbol; kwargs...) = 
      xscal1pbasis(P, idxsyms, GetVal{valsym}(); kwargs...)

xscal1pbasis(P::TransformedPolys, idxsyms, valsym::Symbol, validx::Int; kwargs...) = 
      xscal1pbasis(P, idxsyms, GetVali{valsym, validx}(); kwargs...)

function xscal1pbasis(P::TransformedPolys, idxsyms, fval; label = "")
   ISYMS = tuple(keys(idxsyms)...)
   rgs = NamedTuple{ISYMS}( tuple([idxsyms[sym] for sym in ISYMS]...) )
   return XScal1pBasis(fval, ISYMS, rgs, P, label)
end 

function XScal1pBasis(fval, ISYMS::NTuple{NI, Symbol}, 
                      rgs::TRG, P::TransformedPolys{T, TT, TJ}, 
                      label::String = "", 
                      spec = NamedTuple{ISYMS, NTuple{NI, Int}}[], 
                      coeffs = Matrix{T}(undef, (0,0))
                     ) where {NI, T, TT, TJ, TRG}
   return XScal1pBasis{ISYMS, typeof(fval), T, TT, TJ, NI, TRG}(P, fval, rgs, spec, coeffs, label)
end


_idxsyms(basis::XScal1pBasis{ISYMS}) where {ISYMS} = ISYMS

getval(X, basis::XScal1pBasis) = getval(X, basis.fval)

getval_d(X, basis::XScal1pBasis) = getval_d(X, basis.fval)


rand_radial(basis::XScal1pBasis) = rand_radial(basis.P)


# ---------------------- Implementation of Scal1pBasis


Base.length(basis::XScal1pBasis) = size(basis.coeffs, 1)

get_spec(basis::XScal1pBasis) = copy(basis.spec)

get_spec(basis::XScal1pBasis, i::Integer) = basis.spec[i] 

function set_spec!(basis::XScal1pBasis, spec)
   basis.spec = identity.(spec)
   basis.coeffs = zeros(eltype(basis.coeffs), length(spec), length(basis.P))
   return basis 
end

function isadmissible(b::NamedTuple{BSYMS}, basis::XScal1pBasis) where {BSYMS} 
   ISYMS = _idxsyms(basis)
   @assert all(sym in BSYMS for sym in ISYMS)
   return all(b[sym] in basis.rgs[sym] for sym in ISYMS) 
end 

function sparsify!(basis::XScal1pBasis, spec)
   # spec is the part of the basis we keep 
   inds = Vector{Int}(undef, length(spec))
   for (ib, b) in enumerate(spec)
      iold = findall(isequal(b), basis.spec)
      @assert length(iold) == 1
      inds[ib] = iold[1] 
   end
   # keep the original order of the basis since we assume it is ordered
   # by some sensible notion of degree. 
   p = sortperm(inds)
   basis.spec = spec[p]
   basis.coeffs = basis.coeffs[p, :]
   return basis
end
   

degree(b::NamedTuple, basis::XScal1pBasis) = 
         getproperty(b, _idxsyms(basis)[1]) - 1

degree(b::NamedTuple, basis::XScal1pBasis, weight::Dict) = 
      weight[_idxsyms(basis)[1]] * degree(b, basis)
         

"""
`fill_rand_coeffs!(basis::XScal1pBasis, f::Function)`

This fills the parameters of the XScal1pBasis with coefficients generated 
by the function f. We say "random" because most typically, we expect 
`f = rand` or `f = randn` or similar.
"""
function fill_rand_coeffs!(basis::XScal1pBasis, f::Function)
   for n = 1:length(basis.coeffs)
      basis.coeffs[n] = f()
   end
   return basis 
end

"""
`fill_diag_coeffs!(basis::XScal1pBasis, sym = _idxsys(basis)[1])`

This sets XScal1p basis paramerers as follows: 
```
P_{n,v} = J_n
```
where `sym = :n` and `v` is the rest of the symbols joined together. 
I.e., this reduces the XScal basis to a standard Scal basis. 
"""
function fill_diag_coeffs!(basis::XScal1pBasis, sym = _idxsyms(basis)[1])
   fill!(basis.coeffs, 0)
   for (ib, b) in enumerate(basis.spec)
      n = b[sym]
      basis.coeffs[ib, n] = 1
   end
   return basis
end

# ========================= 


==(P1::XScal1pBasis, P2::XScal1pBasis) = _allfieldsequal(P1, P2)


function write_dict(basis::XScal1pBasis{T}) where {T} 
   ISYMS = _idxsyms(basis)
   return Dict(
      "__id__" => "ACE_XScal1pBasis",
          "P" => write_dict(basis.P), 
          "fval" => write_dict(basis.fval), 
          "syms" => [string.(ISYMS)...], 
          "rgs" => Dict([ string(sym) => write_dict(collect(basis.rgs[sym])) 
                          for sym in ISYMS]... ),
          "spec" => convert.(Dict, basis.spec), 
          "coeffs" => write_dict(basis.coeffs), 
          "label" => basis.label 
      )
end

using NamedTupleTools: namedtuple 

function read_dict(::Val{:ACE_XScal1pBasis}, D::Dict) 
   P = read_dict(D["P"])
   fval = read_dict(D["fval"])
   ISYMS = tuple(Symbol.(D["syms"])...)
   rgs = namedtuple(Dict( [  sym => read_dict(D["rgs"][string(sym)]) 
                             for sym in ISYMS ]... )) |> NamedTuple{ISYMS}
   spec = NamedTuple{ISYMS}.(namedtuple.(D["spec"]))
   coeffs = read_dict(D["coeffs"])
   return XScal1pBasis(fval, ISYMS, rgs, P, D["label"], spec, coeffs)
end

valtype(basis::XScal1pBasis) = 
      promote_type(valtype(basis.P), eltype(basis.coeffs))

valtype(basis::XScal1pBasis, cfg::AbstractConfiguration) =
      valtype(basis, zero(eltype(cfg)))

valtype(basis::XScal1pBasis, X::AbstractState) = 
      promote_type(valtype(basis.P, getval(X, basis)), eltype(basis.coeffs))


gradtype(basis::XScal1pBasis, X::AbstractState) = 
      dstate_type(valtype(basis, X), X)



argsyms(basis::XScal1pBasis) = ( _varsym(basis), )

symbols(basis::XScal1pBasis) = [ _idxsyms(basis)... ]

indexrange(basis::XScal1pBasis) = basis.rgs

# this is needed to generate the product 1p basis
function get_index(basis::XScal1pBasis, b::NamedTuple) 
   ISYMS = _idxsyms(basis)
   b1 = select(b, ISYMS)
   idx = findall(isequal(b1), basis.spec)
   if length(idx) != 1
      @show b1 
      @show basis.spec 
   end
   if length(idx) == 0
      @error("XScal1pBasis : didn't find b in the spec")
      @show b 
      @show basis.spec 
      error("didn't find b in the spec")
   elseif length(idx) > 1
      error("b appears in spec more than once")
   end
   return idx[1]
end 


# ---------------------------  Evaluation code
#


evaluate!(B, basis::XScal1pBasis, X::AbstractState) =
      evaluate!(B, basis, getval(X, basis))

function evaluate!(B, basis::XScal1pBasis, x::Number)
   P = evaluate(basis.P, x)
   mul!(B, basis.coeffs, P)
   return B 
end 


function evaluate_d!(dB, basis::XScal1pBasis, X::AbstractState)
   TDX = eltype(dB)
   x = getval(X, basis)
   dP = acquire_dB!(basis.P, x)
   evaluate_d!(dP, basis.P, x)
   dx = getval_d(X, basis)
   dB[:] .= TDX.( Ref(dx) .* (basis.coeffs * dP) )
   release_dB!(basis.P, dP)
   return dB
end

function evaluate_ed!(B, dB, basis::XScal1pBasis, X::AbstractState)
   TDX = eltype(dB)
   x = getval(X, basis)
   P = acquire_B!(basis.P, x)
   dP = acquire_dB!(basis.P, x)
   evaluate!(P, basis.P, x)
   evaluate_d!(dP, basis.P, x)
   mul!(B, basis.coeffs, P)
   dx = getval_d(X, basis)
   dB[:] .= TDX.( Ref(dx) .* (basis.coeffs * dP) )
   release_B!(basis.P, P)
   release_dB!(basis.P, dP)
   return B, dB
end


# *** TODO 
# # this one we probably only need for training so can relax the efficiency a bit 
# function evaluate_dd(basis::Scal1pBasis, X::AbstractState) 
#    ddP = ForwardDiff.derivative(x -> evaluate_d(basis, _val(X, basis)))
#    TDX = gradtype(basis, X)
#    return _scal1pbasis_grad.(Ref(TDX), Ref(basis), ddP_n)
# end


#=   *** TODO 
# -------------- AD codes 

import ChainRules: rrule, ZeroTangent, NoTangent

function _rrule_evaluate(basis::Scal1pBasis, X::AbstractState, 
                         w::AbstractVector{<: Number})
   @assert _varidx(basis) == 1
   x = _val(X, basis)
   a = _rrule_evaluate(basis.P, x, real.(w))
   TDX = ACE.dstate_type(a, X)
   return TDX( NamedTuple{(_varsym(basis),)}( (a,) ) )
end

rrule(::typeof(evaluate), basis::Scal1pBasis, X::AbstractState) = 
                  evaluate(basis, X), 
                  w -> (NoTangent(), NoTangent(), _rrule_evaluate(basis, X, w))

             
                  
function _rrule_evaluate_d(basis::Scal1pBasis, X::AbstractState, 
                           w::AbstractVector)
   @assert _varidx(basis) == 1
   x = _val(X, basis)
   w1 = [ _val(w, basis) for w in w ]
   a = _rrule_evaluate_d(basis.P, x, w1)
   TDX = ACE.dstate_type(a, X)
   return TDX( NamedTuple{(_varsym(basis),)}( (a,) ) )
end

function rrule(::typeof(evaluate_d), basis::Scal1pBasis, X::AbstractState)
   @assert _varidx(basis) == 1
   x = _val(X, basis)
   dB_ = evaluate_d(basis.P, x)
   TDX = dstate_type(valtype(basis, X), X)
   dB = [ TDX( NamedTuple{(_varsym(basis),)}( (dx,) ) )  for dx in dB_ ]
   return dB, 
          w -> (NoTangent(), NoTangent(), _rrule_evaluate_d(basis, X, w))
end

=#
