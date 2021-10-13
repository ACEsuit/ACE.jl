
##

using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
           Product1pBasis, Scal1pBasis
using Random: shuffle
using ACEbase.Testing: fdtest, print_tf

##

maxdeg = 5
r0 = 1.0 
rcut = 3.0 
maxorder = 3
Bsel = SimpleSparseBasis(maxorder, maxdeg)

trans = PolyTransform(1, r0)
Pk = ACE.scal1pbasis(:x, :k, maxdeg, trans, rcut)
RnYlm = ACE.Utils.RnYlm_1pbasis()

B1p = RnYlm * Pk
ACE.init1pspec!(B1p, Bsel) 
length(B1p)

##

PosScalState{T} = ACE.State{(:rr, :x), Tuple{SVector{3, T}, T}}

Base.promote_rule(::Union{Type{S}, Type{PosScalState{S}}}, 
             ::Type{PosScalState{T}}) where {S, T} = 
      PosScalState{promote_type(S, T)}

X = rand(PosScalState{Float64})
cfg = ACEConfig([ rand(PosScalState{Float64}) for _=1:10 ])

Rn = B1p.bases[1]
Ylm = B1p.bases[2]
Pk = B1p.bases[3]

##

for (B, sym, notsym) in ((Pk, Rn, Ylm), 
                         (:x, :rr, :rr), 
                         (:rr, :x, :x) )
   dB1 = evaluate_d(B, X)
   @which evaluate_d(B, X, sym)
   dB2 = evaluate_d(B, X, sym)
   dB3 = evaluate_d(B, X, notsym)
   dB4 = evaluate_d(B, X, :bob)
   println(@test(dB1 == dB2))
   all(iszero, norm.(dB3))
   all(iszero, norm.(dB4))
end

##
