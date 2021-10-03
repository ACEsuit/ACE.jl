using LinearAlgebra: length
using ACE, ACEbase, Test, ACE.Testing
using ACE: evaluate, SymmetricBasis, PIBasis, O3, State
using StaticArrays
using ChainRules
import ChainRulesCore: rrule, NoTangent
using Zygote
using Printf, LinearAlgebra #for the fdtestMatrix

##

@info("differentiable model test")

# construct the basis
maxdeg = 6
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, O3(), Bsel)

# # generate a random configuration
nX = 10
cfg = ACEConfig([State(rr = rand(SVector{3, Float64})) for _ in 1:nX])

#initialize the model
np = 2
c_m = rand(SVector{np,Float64}, length(basis))
model = ACE.LinearACEModel(basis, c_m, evaluator = :standard)

##

#define the site energy and site forces calculators
#for this code we define "model" globaly to make the code easier
#to understand (no derivatives according to model objects).
#However, for the full implementation it should be there. 

#a hack to implement .. meaning iterating twice over something
getprops(x) = getproperty.(x, :val)

#calculates the site energy
function eval_model(θ, cfg)
   ACE.set_params!(model, θ)
   return getprops(evaluate(model, cfg))
end


#calculates the adjoing/pullback 
function adj_eval_model(dp, θ, cfg)
   ACE.set_params!(model, θ)
   gp = getprops.(ACE.grad_params(model, cfg))
   gp1 = [ gp[i] .* dp for i = 1:length(gp) ]
   g_cfg = ACE._rrule_evaluate(dp, model, cfg)
   return (NoTangent(), gp1, g_cfg)
end

function ChainRules.rrule(::typeof(eval_model), θ, cfg)
   return eval_model(θ, cfg), dp -> adj_eval_model(dp, θ, cfg) 
end


##


θ = randn(np * length(basis)) ./ (1:(np*length(basis))).^2
c = reinterpret(SVector{2, Float64}, θ)

eval_model(c, cfg)

# FS = props -> sum([ 0.77^n * (1 + props[n]^2)^(1/n) for n = 1:length(props) ] )
FS = props -> sum( (1 .+ props.^2).^0.5 )
fsmodel = cfg -> FS(eval_model(c, cfg))
fsmodel(cfg)

g = Zygote.gradient(fsmodel, cfg)[1]

##

@info("Check the AD Forces for an FS-like model")
Us = randn(SVector{3, Float64}, length(cfg))
F = t -> fsmodel(ACEConfig(cfg.Xs + t * Us))
dF = t -> sum( dot(u, g.rr) for (u,g) in zip(Us, Zygote.gradient(fsmodel, cfg)[1]) )
dF(0.0)

ACEbase.Testing.fdtest(F, dF, 0.0, verbose=true)

##

mat2svecs(M) = collect(reinterpret(SVector{np, Float64}, M))
svecs2vec(M) = collect(reinterpret(Float64, M))

@info("Check grad w.r.t. Params of FS-like model")

fsmodelp = θ -> FS(eval_model( mat2svecs(θ), cfg))
grad_fsmodelp = θ -> Zygote.gradient(c -> FS(eval_model(c, cfg)), 
                                     mat2svecs(θ))[1] |> svecs2vec
grad_fsmodelp(θ)

ACEbase.Testing.fdtest(fsmodelp, grad_fsmodelp, θ)


##

#chainrule for derivative of forces according to parameters
function ChainRules.rrule(::typeof(adj_eval_model), dp, θ, cfg)
   
   # dp should be a vector of the same length as the number of properties

   function secondAdj(dq_)
      # adj = (NoTangent(), gp1, g_cfg) 
      # here we assume that only g_cfg was used, which means that 
      # dq_[3] = force-like vector and dq_[2] == NoTangent() 
      @assert dq_[1] == dq_[2] == Zygote.ZeroTangent()
      @assert dq_[3] isa AbstractVector{<: ACE.DState}
      @assert length(dq_[3]) == length(cfg)
      
      # adj_n = ∑_j dq_j ⋅ ∂B_k / ∂r_j * θ_nk
      # dp ⋅ adj = ∑_n ∑_j dq_j ⋅ ∂B_k / ∂r_j * θ_nk * dp_n 

      dq = dq_[3]  # Vector of DStates
      ACE.set_params!(model, θ)
      # grad[k] = ∑_j dq_j ⋅ ∂B_k / ∂r_j
      grad = ACE.adjoint_EVAL_D1(model, model.evaluator, cfg, dq)

      # gradient w.r.t θ: 
      sdp = SVector(dp...)
      grad_θ = grad .* Ref(sdp)

      # gradient w.r.t. dp 
      grad_dp = sum( θ[k] * grad[k] for k = 1:length(grad) )  |> Vector 

      return NoTangent(), grad_dp, grad_θ, NoTangent()
   end

   return adj_eval_model(dp, θ, cfg), secondAdj
end

##

fsmodel1 = FS ∘ eval_model
grad_fsmodel1 = (c, cfg) -> Zygote.gradient(x -> fsmodel1(c, x), cfg)[1]

y = randn(SVector{3, Float64}, length(cfg))
loss1 = c -> sum(sum(abs2, g.rr - y) for (g, y) in zip(grad_fsmodel1(c, cfg), y))

loss1(c)
g = Zygote.gradient(loss1, c)[1]

F = θ -> loss1( mat2svecs(θ) )
dF = θ -> Zygote.gradient(loss1, mat2svecs(θ))[1] |> svecs2vec

dF(θ)

ACEbase.Testing.fdtest(F, dF, θ; verbose=true)


##
