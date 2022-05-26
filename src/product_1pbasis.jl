using NamedTupleTools: namedtuple, merge

# -------------- Implementation of Product Basis

struct Product1pBasis{NB, TB <: Tuple} <: OneParticleBasis{Any}
   bases::TB
   indices::Vector{NTuple{NB, Int}}
end

function Product1pBasis(bases)
   NB = length(bases)
   return Product1pBasis(bases, NTuple{NB, Int}[]) 
end


import Base.*
*(B1::OneParticleBasis, B2::OneParticleBasis) =
      Product1pBasis((B1, B2))
*(B1::Product1pBasis, B2::OneParticleBasis) =
      Product1pBasis((B1.bases..., B2))
*(B1::OneParticleBasis, B2::Product1pBasis) =
      Product1pBasis((B1, B2.bases...))
*(B1::Product1pBasis, B2::Product1pBasis) =
      Product1pBasis((B1.bases..., B2.bases...))
*(B1::Product1pBasis, B2::B1pComponent) =
      Product1pBasis((B1.bases..., B2))


_numb(b::Product1pBasis{NB}) where {NB} = NB

Base.length(basis::Product1pBasis) = length(basis.indices)


function Base.show(io::IO, basis::Product1pBasis)
   print(io, "Product1pBasis") 
   print(io, basis.bases)
end

Base.getindex(basis::Product1pBasis, i::Integer) = basis.bases[i] 

function Base.getindex(basis::Product1pBasis, label::AbstractString)
   inds = findall(getlabel.(basis.bases) .== label) 
   if length(inds) == 0
      error("label not found amongst 1p basis components")
   elseif length(inds) > 1 
      error("label not unique amongst 1p basis components")
   end
   return basis.bases[inds[1]]
end

# ------------------------- FIO CODES

==(B1::Product1pBasis, B2::Product1pBasis) = 
      ( all(B1.bases .== B2.bases) && 
        B1.indices == B2.indices )

write_dict(B::Product1pBasis) = 
      Dict("__id__" => "ACE_Product1pBasis", 
            "bases" => write_dict.(B.bases), 
          "indices" => B.indices )

function read_dict(::Val{:ACE_Product1pBasis}, D::Dict)
   bases = tuple( read_dict.(D["bases"])... )
   indices = [ tuple(v...) for v in D["indices"] ]
   return Product1pBasis(bases, indices)   
end


# ----------------- evaluation of the basis 

import Base.Cartesian: @nexprs

function _write_A_code(VA, NB)
   prodBi = Meta.parse("B_1[ϕ[1]]" * prod(" * B_$i[ϕ[$i]]" for i = 2:NB))
   if VA == Nothing 
      getVT = "promote_type(" * prod("eltype(B_$i), " for i = 1:NB) * ")"
      getA = Meta.parse("_A = zeros($(getVT), length(basis))")
   else 
      getA = :(_A = A)
   end
   return prodBi, getA 
end


"""
`add_into_A!` : this is an internal function implementing the main evaluation 
for the one-particle basis and possibly add it into the A basis. 
   
There are two ways to call it. 
* Use `A = nothing` for the first argument to allocate the necessary memory to 
evaluate the 1p basis into it.
* Use `A::Vector{T}` to evaluate the 1p basis and add it into `A` directly 
without additional allocation. 
"""
@generated function add_into_A!(A::VA, basis::Product1pBasis{NB}, X) where {NB, VA}
   prodBi, getA = _write_A_code(VA, NB)
   quote
      # evaluate the 1p basis components 
      @nexprs $NB i -> begin 
         bas_i = basis.bases[i]
         B_i = evaluate(bas_i, X)
      end 
      # allocate A if necessary or just name _A = A if A is a buffer 
      $(getA)
      # evaluate the 1p product basis functions and add/write into _A
      for (iA, ϕ) in enumerate(basis.indices)
         @inbounds _A[iA] += $prodBi 
      end
      # release the memory allocated by the 1p basis components they normally 
      # use preallocated chache to avoid too many small allocations.
      @nexprs $NB i -> release!(B_i)
      return _A
   end
end

evaluate(basis::Product1pBasis, X::AbstractState) = 
      add_into_A!(nothing, basis, X)

function evaluate(basis::Product1pBasis, cfg::UConfig)
   @assert length(cfg) > 0 "Product1pBasis can only be evaluated with non-empty configurations"
   # evaluate the first item "manually", then so we know the output types 
   # but then write directly into the allocated array to avoid additional 
   # allocations. 
   A = evaluate(basis, first(cfg))
   for (i, X) in enumerate(cfg)
      i == 1 && continue; 
      add_into_A!(A, basis, X)
   end
   return A 
end 

function evaluate!(A::AbstractVector, basis::Product1pBasis, X::AbstractState)
   fill!(A, zero(eltype(A)))
   add_into_A!(A, basis, X)
   return A
end

function evaluate!(A, basis::Product1pBasis, cfg::UConfig)
   fill!(A, zero(eltype(A)))
   for X in cfg 
      add_into_A!(A, basis, X)
   end
   return A
end


# -------------------- jacobian codes, forward rule  

function _write_dA_code(VDA, NB)
   if VDA == Nothing 
      getDVT = "promote_type(eltype(_A), " * prod("eltype_dB_$i, " for i = 1:NB) * ")"
      getdA = Meta.parse("_dA = zeros($(getDVT), length(basis))")
   else 
      getdA = :(_dA = dA)
   end

end


# args... could be a symbol to enable partial derivatives 
# at the moment this is just passed into evaluate_ed!(...) 
# to not evaluate 1p basis derivatives that aren't needed, but 
# in the future a more complex construction could be envisioned that 
# might considerably reduce the computational cost here... 
@generated function _add_into_A_dA!(A::VA, dA::VDA, basis::Product1pBasis{NB}, X,
                                    args...) where {VA, VDA, NB}
   prodBi, getA = _write_A_code(VA, NB)
   getdA = _write_dA_code(VDA, NB)
   quote
      Base.Cartesian.@nexprs($NB, i -> begin   # for i = 1:NB
         bas_i = basis.bases[i] 
         if !(bas_i isa Discrete1pBasis)
            # only evaluate basis gradients for a continuous basis
            B_i, dB_i = evaluate_ed(bas_i, X, args...)
            eltype_dB_i = eltype(dB_i)
         else
            # we still need the basis values for the discrete basis though
            # TODO: maybe the d part should be a no-op and remove this 
            # case distinction ... 
            B_i = evaluate(bas_i, X)
            eltype_dB_i = Bool 
         end
      end)
      # allocate A if necessary or just name _A = A if A is a buffer 
      $(getA) 
      $(getdA)

      for (iA, ϕ) in enumerate(basis.indices)
         # evaluate A
         @inbounds _A[iA] += $prodBi 

         # evaluate dA
         # TODO: redo this with adjoints!!!!
         #     also reverse order of operations to make fewer multiplications!
         _dA[iA] = zero(eltype(_dA))
         Base.Cartesian.@nexprs($NB, a -> begin  # for a = 1:NB
            if !(basis.bases[a] isa Discrete1pBasis)
               dt = dB_a[ϕ[a]]
               Base.Cartesian.@nexprs($NB, b -> begin  # for b = 1:NB
                  if b != a
                     dt *= B_b[ϕ[b]]
                  end
               end)
               _dA[iA] += dt
            end
         end)
      end
      Base.Cartesian.@nexprs($NB, i -> ( begin   # for i = 1:NB
         # release_B!(bas_i, B_i)
         release!(B_i)
         if !(basis.bases[i] isa Discrete1pBasis)
            release!(dB_i)
         end
      end))
      return _A, _dA 
   end
end


# this is a hack to resolve a method ambiguity. 
add_into_A_dA!(A, dA, basis::Product1pBasis, X) = 
               _add_into_A_dA!(A, dA, basis, X) 
add_into_A_dA!(A, dA, basis::Product1pBasis, X, sym::Symbol) = 
               _add_into_A_dA!(A, dA, basis, X, sym) 


evaluate_ed(basis::Product1pBasis, X::AbstractState) = 
         _add_into_A_dA!(nothing, nothing, basis, X)

function evaluate_ed(basis::Product1pBasis, Xs::UConfig) 
   A, dA1 = evaluate_ed(basis, first(Xs))
   dA = zeros(eltype(dA1), length(basis), length(Xs))
   dA[:, 1] .= dA1[:]
   release!(dA1)
   for (i, X) in enumerate(Xs)
      i == 1 && continue; 
      _add_into_A_dA!(A, (@view dA[:, i]), basis, X)
   end
   return A, dA 
end 

function evaluate_d(basis::Product1pBasis, X::Union{AbstractState, UConfig})
   A, dA = evaluate_ed(basis, X)
   release!(A) 
   return dA 
end

# ------------- Partial derivative functionality 

_check_args_is_sym() = true 
_check_args_is_sym(::Symbol) = true


# args... may be empty or a symbol  for partial derivatives
function evaluate_ed!(A, dA, basis::OneParticleBasis,
                     cfg::UConfig, args...)
   @assert _check_args_is_sym(args...)
   fill!(A, 0)
   for (j, X) in enumerate(cfg)
      add_into_A_dA!(A, (@view dA[:, j]), basis, X, args...)
   end
   return A, dA
end

# args... may be empty or a symbol for partial derivatives
function evaluate_ed!(A, dA, basis::Product1pBasis, X::AbstractState, args...)
   @assert _check_args_is_sym(args...)
   fill!(A, 0)
   add_into_A_dA!(A, dA, basis, X, args...)
   return A, dA
end

# ----------------------------------------

_symbols_prod(bases) = tuple(union( symbols.(bases)... )...)

symbols(basis::Product1pBasis) = _symbols_prod(basis.bases)

function indexrange(basis::Product1pBasis)
   allsyms = tuple(symbols(basis)...)
   rg = Dict{Symbol, Vector{Any}}([ sym => [] for sym in allsyms]...)
   for b in basis.bases
      rgb = indexrange(b)
      for sym in allsyms
         if haskey(rgb, sym)
            rg[sym] = union(rg[sym], rgb[sym])
         end
      end
   end
   # HACK: fix the m range based on the maximal l-range
   #       this needs to be suitably generalised if we have multiple
   #       (l, m) pairs, e.g. (l1, m1), (l2, m2)
   if haskey(rg, :m)
      maxl = maximum(rg[:l])
      rg[:m] = collect(-maxl:maxl)
   end

   # convert the range into a named tuple so that we remember the order!!
   return NamedTuple{allsyms}(ntuple(i -> rg[allsyms[i]], length(allsyms)))
end

isadmissible(b, basis::Product1pBasis) = all(isadmissible.(Ref(b), basis.bases))

function set_spec!(basis::Product1pBasis{NB}, spec) where {NB}
   empty!(basis.indices)
   for b in spec
      inds = ntuple(i -> get_index(basis.bases[i], b), NB)
      push!(basis.indices, inds)
   end
   return basis
end

get_spec(basis::Product1pBasis) = [ get_spec(basis, i) for i = 1:length(basis) ]

function get_spec(basis::Product1pBasis, i::Integer) 
   inds = basis.indices[i] 
   specs = get_spec.(basis.bases, inds)
   # TODO: here we should check that we are only merging compatible tuples, 
   #       e.g. (n = 5, l = 2), (l = 2, m = -1) is ok 
   #       but  (n = 5, l = 2), (l = 3, m = -1) is forbidden!
   return merge(specs...)
end

degree(b, basis::Product1pBasis) = sum( degree(b, B) for B in basis.bases )

degree(b::NamedTuple, basis::Product1pBasis, weight::Dict) = 
      sum( degree(b, B, weight) for B in basis.bases )

# TODO: this looks like a horrible hack ...
function rand_radial(basis::Product1pBasis)
   for B in basis.bases
      if B isa ScalarACEBasis
         return rand_radial(B)
      end
   end
   return nothing
end

# -------------- sparsification 

function sparsify!(basis1p::Product1pBasis, keep::AbstractVector{<: NamedTuple})
   # spec, keep, new_spec will be lists of named tuples, 
   #                    e.g. [ (n = , l = , m = ), ... ]
   spec = get_spec(basis1p)
   new_spec = eltype(spec)[]
   new_inds = Vector{Int}(undef, length(spec))
   for (ib, b) in enumerate(spec)
      if b in keep 
         push!(new_spec, b)
         new_inds[ib] = length(new_spec)
      end
   end

   # now we need to recompute the indices array, this can be easily done via 
   # set_spec!(basis::Product1pBasis{NB}, spec), but before we do that 
   # we should sparsify the basis components as well 
   #  .... but it is not so clear that his is a good idea, maybe the 
   #       1p basis components should just remain frozen???
   #       => turn this off for now 
   # TODO - return to this point?!?!?
   # for bas_i in basis1p.bases 
   #    _sparsify_component!(bas_i, new_spec)
   # end

   # finally fix the basis1pspec internally: 
   set_spec!(basis1p, new_spec)

   # return the old to new index mapping so that the pibasis can fix itself. 
   return basis1p, new_inds
end


using NamedTupleTools: select 

# """
# this performs some generic work to sparsify a 1p-basis component. 
# but the actual sparsificatin happens in the individual basis implementations 
# """
# function _sparsify_component!(basis1p, keep)
#    # if basis1p has no symbols (e.g. a multiplier) then it means it must 
#    # be a one-component basis, so there is nothing to sparsify.
#    syms = symbols(basis1p)
#    if isempty(syms)
#       return basis1p
#    end
#    # get rid of all info we don't need 
#    keep1 = unique( select.(keep, Ref(syms)) )
#    # double-check that keep1 is compatible 
#    spec = get_spec(basis1p) 
#    @assert all(b in spec for b in keep1)
#    # now get the basis spec and get the list of indices to keep 
#    if length(keep1) < length(spec)
#       # Ikeep = findall( [b in keep1 for b in spec] )
#       # sparsify!(basis1p, Ikeep)
#       sparsify!(basis1p, keep1)
#    end 
#    return basis1p 
# end


# --------------- AD codes

# import ChainRules: rrule, NoTangent, ZeroTangent

# _evaluate_bases(basis::Product1pBasis{NB}, X::AbstractState) where {NB} = 
#       ntuple(i -> evaluate(basis.bases[i], X), NB)

# _evaluate_A(basis::Product1pBasis{NB}, BB) where {NB} = 
#       [ prod(BB[i][ϕ[i]] for i = 1:NB) for ϕ in basis.indices ]

# evaluate(basis::Product1pBasis, X::AbstractState) = 
#       _evaluate_A(basis, _evaluate_bases(basis, X)) 

# function _rrule_evaluate(basis::Product1pBasis{NB}, X::AbstractState, 
#                          w::AbstractVector{<: Number}, 
#                          BB = _evaluate_bases(basis, X)) where {NB}
#    VT = promote_type(valtype(basis, X), eltype(w))

#    # dB = evaluate_d(basis, X)
#    # return sum( (real(w) * real(db) + imag(w) * imag(db)) 
#    #             for (w, db) in zip(w, dB) )

#    # Compute the differentials for the individual sub-bases 
#    Wsub = ntuple(i -> zeros(VT, length(BB[i])), NB) 
#    for (ivv, vv) in enumerate(basis.indices)
#       for t = 1:NB 
#          _A = one(VT)
#          for s = 1:NB 
#             if s != t 
#                _A *= BB[s][vv[s]]
#             end
#          end
#          Wsub[t][vv[t]] += w[ivv] * conj(_A)
#       end
#    end

#    # now these can be propagated into the inner basis 
#    #  -> type instab to be fixed here 
#    g = sum( _rrule_evaluate(basis.bases[t], X, Wsub[t] )
#             for t = 1:NB )
#    return g
# end

# function rrule(::typeof(evaluate), basis::Product1pBasis, X::AbstractState)
#    BB = _evaluate_bases(basis, X)
#    A = _evaluate_A(basis, BB)
#    return A, 
#       w -> (NoTangent(), NoTangent(), _rrule_evaluate(basis, X, w, BB))
# end


#    function _rrule_evaluate(basis::Scal1pBasis, X::AbstractState, 
#       w::AbstractVector{<: Number})
# x = _val(X, basis)
# a = _rrule_evaluate(basis.P, x, w)
# TDX = ACE.dstate_type(a, X)
# return TDX( NamedTuple{(_varsym(basis),)}( (a,) ) )
# end
