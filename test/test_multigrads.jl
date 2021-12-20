


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

trans = trans = PolyTransform(1, r0)
Pk = ACE.scal1pbasis(:x, :k, maxdeg, trans, rcut)
RnYlm = ACE.Utils.RnYlm_1pbasis()

B1p = RnYlm * Pk
ACE.init1pspec!(B1p, Bsel) 
length(B1p)

##

PosScalState{T} = ACE.State{NamedTuple{(:rr, :x), Tuple{SVector{3, T}, T}}}

Base.promote_rule(::Union{Type{S}, Type{PosScalState{S}}}, 
             ::Type{PosScalState{T}}) where {S, T} = 
      PosScalState{promote_type(S, T)}

X = rand(PosScalState{Float64})
cfg = ACEConfig([ rand(PosScalState{Float64}) for _=1:10 ])

Rn = B1p.bases[1]
Ylm = B1p.bases[2]
Pk = B1p.bases[3]

ACE.gradtype(B1p, cfg)
ACE.valtype(B1p, cfg)

ACE.acquire_B!(Pk, cfg)

A = evaluate(B1p, cfg)
dA = evaluate_d(B1p, cfg)
A1, dA1 = ACE.evaluate_ed(B1p, cfg)


println_slim(@test( A ≈ A1 ))
println_slim(@test( dA ≈ dA1 ))

##

# gradient test 

_vec2X(x) = PosScalState{eltype(x)}((rr = SVector{3}(x[1:3]), x = x[4]))
_vec2cfg(x) = ACEConfig( [_vec2X(x)] )
_X2vec(X) = [X.rr; [X.x]]

for ntest = 1:30
   x0 = randn(4)
   c = rand(length(B1p))
   F = x -> sum(ACE.evaluate(B1p, _vec2X(x)) .* c)
   dF = x -> _X2vec( sum(ACE.evaluate_d(B1p, _vec2cfg(x)) .* c))
   print_tf(@test fdtest(F, dF, x0; verbose=false))
end
println()


##
