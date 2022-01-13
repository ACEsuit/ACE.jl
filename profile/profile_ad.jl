using ACE, StaticArrays, BenchmarkTools

using Zygote: gradient

Nprop = 2
wL = 1.5
deg = 10
ord = 3

Bsel = SparseBasis(; maxorder = ord, p = 1, default_maxdeg = deg,
                        weight = Dict(:n => 1.0, :l => wL))   
B1p = ACE.Utils.RnYlm_1pbasis(maxdeg = deg, 
                                 maxL = ceil(Int, deg / wL), 
                                 Bsel = Bsel)

φ = ACE.Invariant()
bsis = ACE.SymmetricBasis(φ, B1p, ACE.O3(), Bsel)

#create a multiple property model
W = rand(Nprop, length(bsis))
c = [SVector{size(W)[1]}(W[:,i]) for i in 1:size(W)[2]]
LM = ACE.LinearACEModel(bsis, c, evaluator = :standard)

# generate a configuration
nX = 100
Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
cfg = ACEConfig(Xs)





@info("evaluation")
@btime ACE.evaluate(LM, cfg);

@info("der parameters")
@btime ACE.grad_params(LM, cfg);
@info("calculate gradient")
@btime ACE.grad_config(LM, cfg);
@info("Adjoint gradient")
@btime ACE._rrule_evaluate(ones(Nprop), LM, cfg);
@info("der grad according to params")
@btime ACE.grad_params_config(LM, cfg)
@info("Adjoint der grad according to params")
adj = [ACE.DState(rr=rand(SVector{3, Float64})) for _ = 1:length(cfg)]
@btime ACE.adjoint_EVAL_D(LM, LM.evaluator, cfg, adj)

@info("Zygote calls")

site_energy(m,x) = sum(ACE.val.(ACE.evaluate(m, x)))
@info("sum over properties evaluate")
@btime site_energy(LM, cfg)
@info("der parameters")
@btime gradient(x->site_energy(x,cfg), LM)
@info("calculate gradient")
@btime gradient(x->site_energy(LM,x), cfg)
@info("der params of gradient, sum for adjoint")
@btime gradient(m->sum(sum(gradient(x->site_energy(m,x), cfg)[1]).rr), LM)


@info("define a loss with gradients")
FS = props -> sum( (1 .+ ACE.val.(props).^2).^0.5 )
sqr(x) = x.rr .^ 2

loss(m, x) = (FS(ACE.evaluate(m,x)))^2 + sum(sum(sqr.(gradient(tx->FS(ACE.evaluate(m,tx)), x)[1])))
lossE(m, x) = (FS(ACE.evaluate(m,x)))^2


@info("evaluate loss full loss")
@btime loss(LM, cfg)

@info("only energies")
@btime lossE(LM, cfg)

@info("der full loss")
@btime gradient(m->loss(m,cfg), LM)

@info("der only energies")
@btime gradient(m->lossE(m,cfg), LM);


