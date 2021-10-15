
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

@info("Checking correct differentiation w.r.t. known and unknown symbols")
for (B, sym, notsym) in zip( (Pk, Rn, Ylm), 
                             (:x, :rr, :rr), 
                             (:rr, :x, :x) )
   dB1 = evaluate_d(B, X)
   @which evaluate_d(B, X, sym)
   dB2 = evaluate_d(B, X, sym)
   dB3 = evaluate_d(B, X, notsym)
   dB4 = evaluate_d(B, X, :bob)
   print_tf(@test(dB1 == dB2))
   print_tf(@test(all(iszero, norm.(dB3))))
   print_tf(@test(all(iszero, norm.(dB4))))
end
println()

##

@info("Check how the product 1p basis handles a partial derivative")
dB = evaluate_d(B1p, X)
dB_x = evaluate_d(B1p, X, :x)
dB_rr = evaluate_d(B1p, X, :rr)
println(@test dB ≈ dB_x + dB_rr)

##

@info("Test partial derivatives for a symmetric basis")
basis = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
evaluate(basis, cfg)
dB = evaluate_d(basis, cfg)
dB_x = evaluate_d(basis, cfg, :x)
dB_rr = evaluate_d(basis, cfg, :rr)
println(@test( dB ≈ dB_x + dB_rr ))


## 

# try out a chainrule? 
using Zygote
using ACE: LinearACEModel, evaluate

c = randn(length(basis)) ./ (1:length(basis)).^2
model = LinearACEModel(basis, c)

# make up some features we can feed into the x variable. 
function x_features(Rs)
   f(r) = exp(- r)
   Xi = [ sum(f(norm(Rs[i] - Rs[j])) for j = 1:length(Rs)) - f(0) 
          for i = 1:length(Rs) ]
   Xs = [ ACE.State(rr = rr, x = x) for (rr, x) in zip(Rs, Xi) ]
   return ACEConfig(Xs)
end

eval_model(Rs) = ACE.val(evaluate( model, x_features(Rs) ))

Rs = 2.5 * randn(SVector{3, Float64}, 10)
eval_model(Rs)


##

Zygote.gradient(eval_model, Rs)