using LinearAlgebra: length
using ACE, ACEbase, Test, ACE.Testing
using ACE: evaluate, SymmetricBasis, PIBasis, O3, State, val, grad_config, rand_vec3
using StaticArrays
using ChainRules
import ChainRulesCore: rrule, NoTangent, ZeroTangent
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

# generate a random configuration
nX = 10
cfg = ACEConfig([State(rr = rand_vec3(B1p["Rn"])) for _ in 1:nX])

#initialize the model
np = 2
c_m = rand(SVector{np,Float64}, length(basis))
model = ACE.LinearACEModel(basis, c_m, evaluator = :standard)

evaluate(model, cfg)
grad_config(model, cfg)

##

θ = randn(np * length(basis)) ./ (1:(np*length(basis))).^2
c = reinterpret(SVector{2, Float64}, θ)
ACE.set_params!(model, c)

FS = props -> sum( (1 .+ val.(props).^2).^0.5 )
fsmodel = cfg -> FS(evaluate(model, cfg))

# @info("check the model and gradient evaluate ok")
fsmodel(cfg)

# Zygote.refresh()
g = Zygote.gradient(fsmodel, cfg)[1]


## checks rrule_evaluate

@info("Check the AD Forces for an FS-like model")
Us = randn(SVector{3, Float64}, length(cfg))
F = t -> fsmodel(ACEConfig(cfg.Xs + t * Us))
dF = t -> sum( dot(u, g.rr) for (u,g) in zip(Us, Zygote.gradient(fsmodel, cfg)[1]) )
dF(0.0)

println(@test all( ACEbase.Testing.fdtest(F, dF, 0.0, verbose=true) ))

## checks adjoint w.r.t. params

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

println(@test all( ACEbase.Testing.fdtest(fsmodelp, grad_fsmodelp, θ) ))


## second-order adjoint (cfg and params)
# THIS TEST CURRENTLY THROWS A SEGFAULT
# ... but only if run as part of the test set and not 
#     when run manually ?!?!?

@info("Check AD for a second partial derivative w.r.t cfg and params")


# fsmodel1 = (model, cfg) -> FS(evaluate(model, cfg))
# grad_fsmodel1 = (model, cfg) -> Zygote.gradient(x -> fsmodel1(model, x), cfg)[1]

# y = randn(SVector{3, Float64}, length(cfg))
# loss1 = model -> sum(sum(abs2, g.rr - y) 
#                      for (g, y) in zip(grad_fsmodel1(model, cfg), y))

# # check that loss and gradient evaluate ok 
# loss1(model)
# Zygote.refresh()
# g = Zygote.gradient(loss1, model)[1]  # SEGFAULT IN THIS LINE ON J1.7!!!

# # wrappers to take derivatives w.r.t. the vector or parameters
# F1 = θ -> ( ACE.set_params!(model, mat2svecs(θ)); 
#             loss1(model) )

# dF1 = θ -> ( ACE.set_params!(model, mat2svecs(θ)); 
#              Zygote.gradient(loss1, model)[1] |> svecs2vec  )

# F1(θ)
# dF1(θ)
# println(@test all( ACEbase.Testing.fdtest(F1, dF1, θ; verbose=true) )) 
@warn("test removed due to unexplained segfaults only occuring in testing runs")

