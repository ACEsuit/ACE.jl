using LinearAlgebra: length
using ACE, ACEbase, Test, ACE.Testing
using ACE: evaluate, SymmetricBasis, PIBasis, O3, State
using StaticArrays
using ChainRules
import ChainRulesCore: rrule, NoTangent, ZeroTangent
using Zygote
using Zygote: @thunk 
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

# generate a random configuration
nX = 10
cfg = ACEConfig([State(rr = rand(SVector{3, Float64})) for _ in 1:nX])

#initialize the model
np = 2
c_m = rand(SVector{np,Float64}, length(basis))
model = ACE.LinearACEModel(basis, c_m, evaluator = :standard)

evaluate(model, cfg)

##

# an x -> x.val implementation with custom adjoints to sort out the 
# mess created by the AbstractProperties
# maybe this feels a bit wrong, definitely a hack. What might be nicer 
# is to introduce a "Dual Property" similar to the "DState"; Then we 
# could have something along the lines of  DProp * Prop = scalar or 
# _contract(DProp, Prop) = scalar; That would be the "systematic" and 
# "disciplined" way of implementing this. 

val(x) = x.val 

function _rrule_val(dp, x)     # D/Dx (dp[1] * dx)
   @assert dp isa Number 
   return NoTangent(), dp
end

rrule(::typeof(val), x) = 
         val(x), 
         dp -> _rrule_val(dp, x)

function rrule(::typeof(_rrule_val), dp, x)   # D/D... (0 + dp * dq[2])
      @assert dp isa Number 
      function second_adj(dq)
         @assert dq[1] == ZeroTangent() 
         @assert dq[2] isa Number 
         return NoTangent(), dq[2], ZeroTangent()
      end
      return _rrule_val(dp, x), second_adj
end 

## 

#calculates the adjoing/pullback 
function adj_evaluate(dp, model::ACE.LinearACEModel, cfg)
   # dp = getproperty.(dp_, :val)
   gp_ = ACE.grad_params(model, cfg)
   gp = [ val.(a .* dp) for a in gp_ ]
   g_cfg = ACE._rrule_evaluate(dp, model, cfg) # rrule for cfg only...
   return NoTangent(), gp, g_cfg
end

# this is monkey-patching the rotten rrule inside of ACE
# ... and should replace that rrule. ALso introduce thunks to prevent 
#     evaluating more than we need.
function ChainRules.rrule(::typeof(evaluate), model::ACE.LinearACEModel, cfg::ACEConfig)
   return evaluate(model, cfg), 
          dp -> adj_evaluate(dp, model, cfg)
end


##


θ = randn(np * length(basis)) ./ (1:(np*length(basis))).^2
c = reinterpret(SVector{2, Float64}, θ)
ACE.set_params!(model, c)

evaluate(model, cfg)

# FS = props -> sum([ 0.77^n * (1 + props[n]^2)^(1/n) for n = 1:length(props) ] )
FS = props -> sum( (1 .+ val.(props).^2).^0.5 )
fsmodel = cfg -> FS(evaluate(model, cfg))
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

mat2svecs(M::AbstractArray{T}) where {T} =   
      collect(reinterpret(SVector{np, T}, M))
svecs2vec(M::AbstractVector{<: SVector{N, T}}) where {N, T} = 
      collect(reinterpret(T, M))

@info("Check grad w.r.t. Params of FS-like model")

fsmodelp = θ -> ( ACE.set_params!(model, mat2svecs(θ)); 
                  FS(evaluate(model, cfg)) )
grad_fsmodelp = θ -> (
         ACE.set_params!(model, mat2svecs(θ)); 
         Zygote.gradient(model -> FS(evaluate(model, cfg)), model)[1] |> svecs2vec )
grad_fsmodelp(θ)

ACEbase.Testing.fdtest(fsmodelp, grad_fsmodelp, θ)


##

#chainrule for derivative of forces according to parameters
function ChainRules.rrule(::typeof(adj_evaluate), dp, model::ACE.LinearACEModel, cfg)

   # dp should be a vector of the same length as the number of properties
   # dp = getproperty.(dp_, :val)

   function _second_adj(dq_)
      # adj = (NoTangent(), gp1, g_cfg) 
      # here we assume that only g_cfg was used, which means that 
      # dq_[3] = force-like vector and dq_[2] == NoTangent() 
      @assert dq_[1] == dq_[2] == Zygote.ZeroTangent()
      @assert dq_[3] isa AbstractVector{<: ACE.DState}
      @assert length(dq_[3]) == length(cfg)
      
      # adj_n = ∑_j dq_j ⋅ ∂B_k / ∂r_j * θ_nk
      # dp ⋅ adj = ∑_n ∑_j dq_j ⋅ ∂B_k / ∂r_j * θ_nk * dp_n 

      dq = dq_[3]  # Vector of DStates
      # grad[k] = ∑_j dq_j ⋅ ∂B_k / ∂r_j
      grad = ACE.adjoint_EVAL_D1(model, model.evaluator, cfg, dq)

      # gradient w.r.t θ: 
      sdp = SVector(dp...)
      grad_θ = grad .* Ref(sdp)

      # gradient w.r.t. dp 
      grad_dp = sum( model.c[k] * grad[k] for k = 1:length(grad) )  |> Vector 

      return NoTangent(), grad_dp, grad_θ, NoTangent()
   end

   return adj_evaluate(dp, model, cfg), _second_adj
end


##

fsmodel1 = (model, cfg) -> FS(evaluate(model, cfg))
grad_fsmodel1 = (model, cfg) -> Zygote.gradient(x -> fsmodel1(model, x), cfg)[1]

y = randn(SVector{3, Float64}, length(cfg))
loss1 = model -> sum(sum(abs2, g.rr - y) 
                     for (g, y) in zip(grad_fsmodel1(model, cfg), y))

loss1(model)
g = Zygote.gradient(loss1, model)[1]


F1 = θ -> ( ACE.set_params!(model, mat2svecs(θ)); 
            loss1(model) )

dF1 = θ -> ( ACE.set_params!(model, mat2svecs(θ)); 
            Zygote.gradient(loss1, model)[1] |> svecs2vec  )

F1(θ)
dF1(θ)
ACEbase.Testing.fdtest(F1, dF1, θ; verbose=true)    # TODO: why does this now fail? 


##
