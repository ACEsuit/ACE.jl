#
#  - split off the linear transformation 
#  - remove the abstract oneparticle basis crap and incorporate it into 
#    the Product1pBasis -> renamed as OnepBasis
#  - spec simply specifies in what order the wrapped basis generates the 
#    basis functions, basically don't allow doing it lazy anymore 
#  - P_qa(r) is actually P_q(r, a) so the only issue with this is 
#    if we want different basis length for different species, but even then 
#    we could just fill in zeros
#  - Linear1pTransform: just a linear mapping
#

# MORE NOTES 
#  - consider leaving FVAL in here for now, but move it out later...
#  - changed getval(fval, X) to evaluate(fval, X)
#  - can we get away with only proviging gradients w.r.t. the 
#    symbols used in this component, or do we need the whole thing? 


import NamedTupleTools
using NamedTupleTools: namedtuple


@doc raw"""
`struct B1pComponent:` Wraps a one-particle basis component. 

`basis` is a structure that can compute a basis, i.e. a vector of scalars, 
real or complex. each element of this vector is "specified" by a NamedTuple, 
stored in `spec`. E.g., for an `Rn` type basis the spec is would be 
```julia
   spec = [ (n = 1,), (n=2, ), ... ]
```
while for a Ylm type basis it would be 
```julia 
   spec = [ (l=0, m=0), (l=1, m=-1), (l=1, m=0), ... ]
```

There is no convenience constructor, the `B1pComponent` must be constructed 
"by hand", for which several wrapper functions are written.
"""
struct B1pComponent{ISYMS, TT, TB, FVAL}
   basis::TB
   fval::FVAL
   spec::Vector{NamedTuple{ISYMS, TT}}
   degrees::Vector{Int}
   label::String 
   meta::Dict{String,Any}
   # ------------ derived fields
   invspec::Dict{NamedTuple{ISYMS, TT}, Int}
   # todo - fields for temporary arrays ...  
end

function B1pComponent(basis, fval, spec::AbstractVector{<: NamedTuple}, 
                      degrees::Vector{Int}, 
                      label::AbstractString, 
                      meta = Dict{String, Any}())
   spec1 = collect(spec)                      
   invspec = Dict{eltype(spec1), Int}()
   for (i, b) in enumerate(spec)
      invspec[b] = i 
   end
   return B1pComponent(basis, fval, spec, degrees, label, meta, invspec)
end


# -------------- management of the basis specification, in particular the 
#                interaction with Product1pBasis 

getlabel(basis::B1pComponent) = basis.label

Base.length(basis::B1pComponent) = length(basis.spec)

_idxsyms(basis::B1pComponent{ISYMS}) where {ISYMS} = ISYMS

get_spec(basis::B1pComponent) = copy(basis.spec)

get_spec(basis::B1pComponent, i::Integer) = basis.spec[i] 

# TODO - I don't think this is needed anymore, but the sparsification will 
#        have to be moved in here and something be done about that...
# function set_spec!(basis::XScal1pBasis, spec)
#    basis.spec = identity.(spec)
#    basis.coeffs = zeros(eltype(basis.coeffs), length(spec), length(basis.P))
#    return basis 
# end

function isadmissible(b::NamedTuple{BSYMS}, basis::B1pComponent) where {BSYMS} 
   ISYMS = _idxsyms(basis)
   # this is an assert since it should ALWAYS be true, if not there is a bug
   @assert all(sym in BSYMS for sym in ISYMS)  
   # project to the ISYMS and check it is in the b1pcomponent 
   return haskey(basis.invspec, b[ISYMS])
end

# TODO - LATER 
# function sparsify!(basis::XScal1pBasis, spec)
#    # spec is the part of the basis we keep 
#    inds = Vector{Int}(undef, length(spec))
#    for (ib, b) in enumerate(spec)
#       iold = findall(isequal(b), basis.spec)
#       @assert length(iold) == 1
#       inds[ib] = iold[1] 
#    end
#    # keep the original order of the basis since we assume it is ordered
#    # by some sensible notion of degree. 
#    p = sortperm(inds)
#    basis.spec = spec[p]
#    basis.coeffs = basis.coeffs[p, :]
#    return basis
# end

symbols(basis::B1pComponent) = [ _idxsyms(basis)... ]

function indexrange(basis::B1pComponent) 
   ISYMS = _idxsyms(basis)
   minidx = Dict([sym => minimum(b[sym] for b in basis.spec) for sym in ISYMS]...)
   maxidx = Dict([sym => maximum(b[sym] for b in basis.spec) for sym in ISYMS]...)
   return Dict([sym => minidx[sym]:maxidx[sym] for sym in ISYMS]...)
end

# this is needed to generate the product 1p basis - it returns the 
# index of the 1p basis function component used in the 1p basis function 
# specified by b 
function get_index(basis::B1pComponent, b::NamedTuple) 
   ISYMS = _idxsyms(basis)
   b1 = b[ISYMS]
   if !haskey(basis.invspec, b1)
      error("B1pComponent ($(basis.label) : can't find $(b1) in spec")
   end
   return basis.invspec[b1]
end



# -------------- degree calculations

# this should be revisited, doesn't feel clean yet 

function degree(b::NamedTuple, basis::B1pComponent)
   ISYMS = _idxsyms(basis)
   idx = basis.invspec[ b[ISYMS] ]
   return basis.degrees[idx]
end

function degree(b::NamedTuple, basis::B1pComponent, weight::Dict)
   ISYMS = _idxsyms(basis)
   return weight[ISYMS[1]] * degree(b, basis)
end

# --------------- IO operations 


==(P1::B1pComponent, P2::B1pComponent) = ( 
      (P1.basis == P2.basis) && (P1.spec == P2.spec) && 
      (P1.label == P2.label) && (P1.fval == P2.fval) && 
      (P1.degrees == P2.degrees) )


function write_dict(basis::B1pComponent)
   ISYMS = _idxsyms(basis)
   return Dict("__id__" => "ACE_B1pComponent", 
                 "syms" => [ string.(ISYMS) ...], 
                "basis" => write_dict(basis.basis), 
                 "fval" => write_dict(basis.fval), 
                 "spec" => convert.(Dict, basis.spec), 
              "degrees" => basis.degrees, 
                "label" => basis.label )
end


function read_dict(::Val{:ACE_B1pComponent}, D::Dict)
   basis = read_dict(D["basis"])
   ISYMS = tuple(Symbol.(D["syms"])...)
   spec = NamedTuple{ISYMS}.(namedtuple.(D["spec"]))
   fval = read_dict(D["fval"])
   return B1pComponent(basis, fval, spec, Int.(D["degrees"]), D["label"])
end


function show(io::IO, basis::B1pComponent)
   vsyms = Transforms.get_symbols(basis.fval)
   strvsyms = filter(!isequal(':'), "$vsyms")
   if length(vsyms) == 1 
      # this doesn't work with unicode!?!?!?
      # strvsyms = strvsyms[1:end-2] * ")"
      strvsyms = prod(split(strvsyms, ','))
   end
   print(io, "B1pComponent( $(basis.label)$(strvsyms)")
   for (key, rg) in indexrange(basis)
      print(io, ", $key = $(rg)")
   end
   print(io, ")")
end


# ------------------- preparation for evaluation and managing temporaries 

valtype(basis::B1pComponent) = valtype(basis.basis)

valtype(basis::B1pComponent, X::AbstractState) = 
      valtype(basis.basis, evaluate(basis.fval, X))

function gradtype(basis::B1pComponent, X::AbstractState) 
   x = evaluate(basis.fval, X)
   # gradient type of the inner basis which knows nothing about states 
   TDB = gradtype(basis.basis, x)
   # now we need to incorporate the grad type of fval itself 
   # dx = evaluate_d(basis.fval, X)
   # the gradient will be a product of a TDB times a TDVAL 
   return Transforms.grad_type_dP(TDB, basis.fval, X)
      # dstate_type(valtype(basis, X), X)
end

acquire_B!(basis::B1pComponent, args...) = 
         Vector{valtype(basis, args...)}(undef, length(basis))

release_B!(basis::B1pComponent, args...) = nothing

acquire_dB!(basis::B1pComponent, X::AbstractState) = 
            Vector{gradtype(basis, X)}(undef, length(basis))

release_dB!(basis::B1pComponent, args...) = nothing


# ------------------------ Evaluation code
#                          this is basically an interface for the inner basis

evaluate(basis::B1pComponent, X::AbstractState) = 
         evaluate(basis.basis, evaluate(basis.fval, X))

# evaluate!(B, basis::B1pComponent, X::AbstractState) =
#       evaluate!(B, basis.basis, )

function evaluate_d(basis::B1pComponent, X::AbstractState) 
   B, dB = evaluate_ed(basis, X)
   release!(B)
   return dB 
end 

function evaluate_ed(basis::B1pComponent, X::AbstractState) 
   x = evaluate(basis.fval, X)
   B, dP = evaluate_ed(basis.basis, x)
   dB = rrule_evaluate(dP, basis.fval, X)
   release!(dP)
   return B, dB
end



# this one we probably only need for training so ...
#   - we can relax the efficiency a bit 
#   - we actually never need the evaluate_dd but only the 
#     associated backpropagation operation which may be a bit simpler 
#     to implement. 
# so we will likely remove this entirely.
# evaluate_dd(basis::B1pComponent, X::AbstractState)


# -------------- AD codes 




#=   *** TODO 
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
