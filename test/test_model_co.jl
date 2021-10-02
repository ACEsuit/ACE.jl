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

FS = props -> sum([ 0.77^n * (1 + props[n]^2)^(1/n) for n = 1:length(props) ] )
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

   function secondAdj(dq_)
      # ∑_j ∂B_k / ∂r_j ⋅ dq_j
      dq = dq_[3]  # Vector of DStates
      ACE.set_params!(model, θ)
      grad = ACE.adjoint_EVAL_D1(model, model.evaluator, cfg, dq)

      # gradient w.r.t θ: 
      sdp = SVector(dp...)
      grad_θ = grad .* Ref(sdp)

      # gradient w.r.t. dp 
      grad_dp = sum( θ[k] * grad[k] for k = 1:length(grad) )

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

# F = θ -> loss1( vec2svecs(θ) )
# dF = θ -> Zygote.gradient(loss1, vec2svecs(θ))[1] |> svecs2vec

# dF(θ)

# ACEbase.Testing.fdtest(F, dF, θ; verbose=true)
;

##

#simple loss function with sum over properties and over forces
function loss(θ)
   props = energyModel(θ, cfg)
   #FS = props -> sum( [ 0.77^n * (1 + props[n]^2)^(1/n) for n = 1:length(props) ] )
   FS = props -> sum( [0.77^n for n = 1:length(props)] .* props )
   Ftemp = Zygote.gradient(x -> FS( energyModel(θ, x) ), cfg)[1]
   floss = f -> sum(abs2, f.rr)
   return(abs2(FS(props)) + sum(floss, Ftemp))
end

# g = Zygote.gradient(loss, c_m)[1] sample on how to get the gradient

##

#functions for testing. basically handling SVectors and testing multiple
#properties

function svector2matrix(sv)
   M = zeros(length(sv[1]), length(sv))
   for i in 1:length(sv)
      M[:,i] = sv[i]
   end
   return M
end

function matrix2svector(M)
   sv = [SVector{size(M)[1]}(M[:,i]) for i in 1:size(M)[2]]
   return sv
end

##
@info("FD test forces")

for _ in 1:5
   Us = randn(SVector{3, Float64}, length(cfg))
   FS = props -> sum([ 0.77^n * (1 + props[n]^2)^(1/n) for n = 1:length(props) ] )
   F = t -> FS(energyModel(c_m, ACEConfig(cfg.Xs + t .* Us)))

   function dF(t)
      forces = Zygote.gradient(x->FS(energyModel(c_m, x)), cfg)[1]
      tmp = zeros(size(forces,1))
      for i in 1:size(forces,1)
         tmp[i] = sum([dot(forces[:,j][i].rr, Us[i]) for j in 1:size(forces,2)])
      end
      return tmp
   end

   print_tf(@test ACEbase.Testing.fdtest(F, dF, zeros(length(cfg)), verbose=false))
end
println()


@info("FD test d(forces)")

#using the sin causes an error. Likewise for any function that requires to evaluate
#the value, for ex x^2. However, it works for constant multipliers. 
#for the pullback we evaluate from the outside to the inside, 

#nonlin(x) = sum(2 .* x)
nonlin(x) = sum(sin.(x))

floss = f -> sum(f.rr)
Ftmp = c -> sum(floss, Zygote.gradient(x -> nonlin(energyModel(c, x)), cfg)[1])
dFtmp = c -> Zygote.gradient(Ftmp, c)

for _ in 1:5
   c = randn(np * length(basis))
   F = c -> Ftmp(matrix2svector(reshape(c, np, length(basis))))
   dF = c -> svector2matrix(dFtmp(matrix2svector(reshape(c, np, length(basis))))[1])
   println(@test ACEbase.Testing.fdtest(F, dF, c, verbose=true))
end
println()


@info("loss function test")

for _ in 1:5
   c = randn(np * length(basis))
   F = c -> loss(matrix2svector(reshape(c, np, length(basis))))
   dF = c -> svector2matrix(Zygote.gradient(loss, matrix2svector(reshape(c, np, length(basis))))[1])[:]

   println(@test ACEbase.Testing.fdtest(F, dF, c, verbose=true))
end
println()

