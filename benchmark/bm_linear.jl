using ACE, StaticArrays, BenchmarkTools, Printf

using Zygote: gradient
using ACE: evaluate, evaluate!, evaluate_d, evaluate_d!
using ACEbase: acquire_B!, acquire_dB!

TX = ACE.PositionState{Float64}
B1p = ACE.Utils.RnYlm_1pbasis()
Rn = B1p.bases[1]
cfg = ACEConfig(rand(TX, Rn, 30))

##

degrees = Dict(2 => [7, 10, 12],
               3 => [7, 9, 11] )

Adegrees = [7, 10, 13, 16, 19] 

# degrees = Dict(2 => [7, 12, 17],
#                3 => [7, 11, 15],
#                4 => [7, 10, 13], 
#                5 => [7, 9, 11] )

wL = 1.5 

Nprop = 2

#zygote tests
site_energy(m,x) = sum(ACE.val.(ACE.evaluate(m, x)))

#loss tests
FS = props -> sum( (1 .+ ACE.val.(props).^2).^0.5 )
sqr(x) = x.rr .^ 2

#full loss
loss(m, x) = (FS(ACE.evaluate(m,x)))^2 + sum(sum(sqr.(gradient(tx->FS(ACE.evaluate(m,tx)), x)[1])))
#only energy loss
lossE(m, x) = (FS(ACE.evaluate(m,x)))^2

##

#linear model
Agroup = BenchmarkGroup()
Agroup["set_params!"] = BenchmarkGroup()
Agroup["evaluate"] = BenchmarkGroup()

#only ACE
Agroup["grad_params"] = BenchmarkGroup()
Agroup["grad_config"] = BenchmarkGroup()
Agroup["_rrule_evaluate"] = BenchmarkGroup()
Agroup["grad_params_config"] = BenchmarkGroup()
Agroup["adjoint_EVAL_D1"] = BenchmarkGroup()

#zygote calls
Agroup["site_energy"] = BenchmarkGroup()
Agroup["Zygote_grad_params"] = BenchmarkGroup()
Agroup["Zygote_grad_config"] = BenchmarkGroup()
Agroup["Zygote_grad_params_config"] = BenchmarkGroup()

#loss functions
Agroup["eval_full_loss"] = BenchmarkGroup()
Agroup["eval_energy_loss"] = BenchmarkGroup()
Agroup["der_full_loss"] = BenchmarkGroup()
Agroup["der_energy_loss"] = BenchmarkGroup()


for deg in Adegrees 
   local B1p
   Bsel = SparseBasis(; maxorder = 1, p = 1, default_maxdeg = deg,
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

   Agroup["set_params!"][deg] = @benchmarkable ACE.set_params!($LM, $c)
   Agroup["evaluate"][deg] = @benchmarkable ACE.evaluate($LM, $cfg)
   


   #ACE native calls for derivatives
   Agroup["grad_params"][deg] = @benchmarkable ACE.grad_params($LM, $cfg)
   Agroup["grad_config"][deg] = @benchmarkable ACE.grad_config($LM, $cfg)
   dp = ones(Nprop) #the forces pullback input
   Agroup["_rrule_evaluate"][deg] = @benchmarkable ACE._rrule_evaluate($dp, $LM, $cfg)
   Agroup["grad_params_config"][deg] = @benchmarkable ACE.grad_params_config($LM, $cfg)
   dq = [ACE.DState(rr=rand(SVector{3, Float64})) for _ = 1:length(cfg)] #the adjoint
   Agroup["adjoint_EVAL_D1"][deg] = @benchmarkable ACE.adjoint_EVAL_D1($LM, $LM.evaluator, $cfg, $dq)



   #Zygote calls for derivatives
   Agroup["site_energy"][deg] = @benchmarkable site_energy($LM, $cfg)
   Agroup["Zygote_grad_params"][deg] = @benchmarkable gradient(x->site_energy(x,$cfg), $LM)
   Agroup["Zygote_grad_config"][deg] = @benchmarkable gradient(x->site_energy($LM,x), $cfg)
   #sum over the function so we don't compute the jacobian
   Agroup["Zygote_grad_params_config"][deg] = @benchmarkable gradient(m->sum(sum(gradient(x->site_energy(m,x), $cfg)[1]).rr), $LM)



   #loss function 
   Agroup["eval_full_loss"][deg] = @benchmarkable loss($LM, $cfg)
   Agroup["eval_energy_loss"][deg] = @benchmarkable lossE($LM, $cfg)
   Agroup["der_full_loss"][deg] = @benchmarkable gradient(m->loss(m,$cfg), $LM)
   Agroup["der_energy_loss"][deg] = @benchmarkable gradient(m->lossE(m,$cfg), $LM)


end 

##

#linear model
Bgroup = BenchmarkGroup()
Bgroup["set_params!"] = BenchmarkGroup()
Bgroup["evaluate"] = BenchmarkGroup()

#only ACE
Bgroup["grad_params"] = BenchmarkGroup()
Bgroup["grad_config"] = BenchmarkGroup()
Bgroup["_rrule_evaluate"] = BenchmarkGroup()
Bgroup["grad_params_config"] = BenchmarkGroup()
Bgroup["adjoint_EVAL_D1"] = BenchmarkGroup()

#zygote calls
Bgroup["site_energy"] = BenchmarkGroup()
Bgroup["Zygote_grad_params"] = BenchmarkGroup()
Bgroup["Zygote_grad_config"] = BenchmarkGroup()
Bgroup["Zygote_grad_params_config"] = BenchmarkGroup()

#loss functions
Bgroup["eval_full_loss"] = BenchmarkGroup()
Bgroup["eval_energy_loss"] = BenchmarkGroup()
Bgroup["der_full_loss"] = BenchmarkGroup()
Bgroup["der_energy_loss"] = BenchmarkGroup()


for ord = keys(degrees), deg in degrees[ord]
   local B1p
   Bsel = SparseBasis(; maxorder = 1, p = 1, default_maxdeg = deg,
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

   Bgroup["set_params!"][ord, deg] = @benchmarkable ACE.set_params!($LM, $c)
   Bgroup["evaluate"][ord, deg] = @benchmarkable ACE.evaluate($LM, $cfg)
   


   #ACE native calls for derivatives
   Bgroup["grad_params"][ord, deg] = @benchmarkable ACE.grad_params($LM, $cfg)
   Bgroup["grad_config"][ord, deg] = @benchmarkable ACE.grad_config($LM, $cfg)
   dp = ones(Nprop) #the forces pullback input
   Bgroup["_rrule_evaluate"][ord, deg] = @benchmarkable ACE._rrule_evaluate($dp, $LM, $cfg)
   Bgroup["grad_params_config"][ord, deg] = @benchmarkable ACE.grad_params_config($LM, $cfg)
   dq = [ACE.DState(rr=rand(SVector{3, Float64})) for _ = 1:length(cfg)] #the adjoint
   Bgroup["adjoint_EVAL_D1"][ord, deg] = @benchmarkable ACE.adjoint_EVAL_D1($LM, $LM.evaluator, $cfg, $dq)



   #Zygote calls for derivatives
   Bgroup["site_energy"][ord, deg] = @benchmarkable site_energy($LM, $cfg)
   Bgroup["Zygote_grad_params"][ord, deg] = @benchmarkable gradient(x->site_energy(x,$cfg), $LM)
   Bgroup["Zygote_grad_config"][ord, deg] = @benchmarkable gradient(x->site_energy($LM,x), $cfg)
   #sum over the function so we don't compute the jacobian
   Bgroup["Zygote_grad_params_config"][ord, deg] = @benchmarkable gradient(m->sum(sum(gradient(x->site_energy(m,x), $cfg)[1]).rr), $LM)



   #loss function 
   Bgroup["eval_full_loss"][ord, deg] = @benchmarkable loss($LM, $cfg)
   Bgroup["eval_energy_loss"][ord, deg] = @benchmarkable lossE($LM, $cfg)
   Bgroup["der_full_loss"][ord, deg] = @benchmarkable gradient(m->loss(m,$cfg), $LM)
   Bgroup["der_energy_loss"][ord, deg] = @benchmarkable gradient(m->lossE(m,$cfg), $LM)


end 

##

linear_suite = BenchmarkGroup() 
linear_suite["A"] = Agroup
linear_suite["B"] = Bgroup 

##

@info("Tune")
tune!(linear_suite)

@info("Run")
results = run(linear_suite, verbose = true)   
