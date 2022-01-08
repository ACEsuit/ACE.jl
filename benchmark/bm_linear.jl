using ACE, StaticArrays, BenchmarkTools, Printf

using Zygote: gradient
using ACE: evaluate, evaluate!, evaluate_d, evaluate_d!
using ACEbase: acquire_B!, acquire_dB!

TX = ACE.PositionState{Float64}
B1p = ACE.Utils.RnYlm_1pbasis()
Rn = B1p.bases[1]
cfg = ACEConfig(rand(TX, Rn, 30))

##

# degrees = Dict(2 => [9, 17],
#                3 => [7, 15] )

Pnumprops = [1, 2, 4] 

degrees = Dict(2 => [10, 17],
               3 => [9, 15],
               5 => [7, 11] )

wL = 1.5 

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
#Pgroup for benchmarking different number of properties (P)
#fixed maxorder = 2 and maxdeg = 7

fixdeg = 7
fixord = 2

#linear model
Pgroup = BenchmarkGroup()
Pgroup["set_params!"] = BenchmarkGroup()
Pgroup["evaluate"] = BenchmarkGroup()

#only ACE
Pgroup["grad_params"] = BenchmarkGroup()
Pgroup["grad_config"] = BenchmarkGroup()
Pgroup["_rrule_evaluate"] = BenchmarkGroup()
Pgroup["grad_params_config"] = BenchmarkGroup()
Pgroup["adjoint_EVAL_D"] = BenchmarkGroup()

# #zygote calls
# Pgroup["site_energy"] = BenchmarkGroup()
# Pgroup["Zygote_grad_params"] = BenchmarkGroup()
# Pgroup["Zygote_grad_config"] = BenchmarkGroup()
# Pgroup["Zygote_grad_params_config"] = BenchmarkGroup()

# #loss functions
# Pgroup["eval_full_loss"] = BenchmarkGroup()
# Pgroup["eval_energy_loss"] = BenchmarkGroup()
# Pgroup["der_full_loss"] = BenchmarkGroup()
# Pgroup["der_energy_loss"] = BenchmarkGroup()

for numprops in Pnumprops 
   local B1p
   Bsel = SparseBasis(; maxorder = fixord, p = 1, default_maxdeg = fixdeg,
                        weight = Dict(:n => 1.0, :l => wL))   
   B1p = ACE.Utils.RnYlm_1pbasis(maxdeg = fixdeg, 
                                 maxL = ceil(Int, fixdeg / wL), 
                                 Bsel = Bsel)

   φ = ACE.Invariant()
   bsis = ACE.SymmetricBasis(φ, B1p, ACE.O3(), Bsel)

   #create a multiple property model
   W = rand(numprops, length(bsis))
   c = [SVector{size(W)[1]}(W[:,i]) for i in 1:size(W)[2]]
   LM = ACE.LinearACEModel(bsis, c, evaluator = :standard)

   Pgroup["set_params!"][numprops] = @benchmarkable ACE.set_params!($LM, $c)
   Pgroup["evaluate"][numprops] = @benchmarkable ACE.evaluate($LM, $cfg)
   


   #ACE native calls for derivatives
   Pgroup["grad_params"][numprops] = @benchmarkable ACE.grad_params($LM, $cfg)
   Pgroup["grad_config"][numprops] = @benchmarkable ACE.grad_config($LM, $cfg)
   dp = ones(numprops) #the forces pullback input
   Pgroup["_rrule_evaluate"][numprops] = @benchmarkable ACE._rrule_evaluate($dp, $LM, $cfg)
   Pgroup["grad_params_config"][numprops] = @benchmarkable ACE.grad_params_config($LM, $cfg)
   dq = [ACE.DState(rr=rand(SVector{3, Float64})) for _ = 1:length(cfg)] #the adjoint
   Pgroup["adjoint_EVAL_D"][numprops] = @benchmarkable ACE.adjoint_EVAL_D($LM, $LM.evaluator, $cfg, $dq)



   # #Zygote calls for derivatives
   # Pgroup["site_energy"][numprops] = @benchmarkable site_energy($LM, $cfg)
   # Pgroup["Zygote_grad_params"][numprops] = @benchmarkable gradient(x->site_energy(x,$cfg), $LM)
   # Pgroup["Zygote_grad_config"][numprops] = @benchmarkable gradient(x->site_energy($LM,x), $cfg)
   # #sum over the function so we don't compute the jacobian
   # Pgroup["Zygote_grad_params_config"][numprops] = @benchmarkable gradient(m->sum(sum(gradient(x->site_energy(m,x), $cfg)[1]).rr), $LM)



   # #loss function 
   # Pgroup["eval_full_loss"][numprops] = @benchmarkable loss($LM, $cfg)
   # Pgroup["eval_energy_loss"][numprops] = @benchmarkable lossE($LM, $cfg)
   # Pgroup["der_full_loss"][numprops] = @benchmarkable gradient(m->loss(m,$cfg), $LM)
   # Pgroup["der_energy_loss"][numprops] = @benchmarkable gradient(m->lossE(m,$cfg), $LM)


end 

##
#Benchmarking derivatives for different sizes of basis (B)
#the number of properties is fixed to 2

Nprop = 2

#linear model
Bgroup = BenchmarkGroup()
Bgroup["set_params!"] = BenchmarkGroup()
Bgroup["evaluate"] = BenchmarkGroup()

#only ACE
Bgroup["grad_params"] = BenchmarkGroup()
Bgroup["grad_config"] = BenchmarkGroup()
Bgroup["_rrule_evaluate"] = BenchmarkGroup()
Bgroup["grad_params_config"] = BenchmarkGroup()
Bgroup["adjoint_EVAL_D"] = BenchmarkGroup()

# #zygote calls
# Bgroup["site_energy"] = BenchmarkGroup()
# Bgroup["Zygote_grad_params"] = BenchmarkGroup()
# Bgroup["Zygote_grad_config"] = BenchmarkGroup()
# Bgroup["Zygote_grad_params_config"] = BenchmarkGroup()

# #loss functions
# Bgroup["eval_full_loss"] = BenchmarkGroup()
# Bgroup["eval_energy_loss"] = BenchmarkGroup()
# Bgroup["der_full_loss"] = BenchmarkGroup()
# Bgroup["der_energy_loss"] = BenchmarkGroup()


for ord = keys(degrees), deg in degrees[ord]
   local B1p
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

   Bgroup["set_params!"][ord, deg] = @benchmarkable ACE.set_params!($LM, $c)
   Bgroup["evaluate"][ord, deg] = @benchmarkable ACE.evaluate($LM, $cfg)
   


   #ACE native calls for derivatives
   Bgroup["grad_params"][ord, deg] = @benchmarkable ACE.grad_params($LM, $cfg)
   Bgroup["grad_config"][ord, deg] = @benchmarkable ACE.grad_config($LM, $cfg)
   dp = ones(Nprop) #the forces pullback input
   Bgroup["_rrule_evaluate"][ord, deg] = @benchmarkable ACE._rrule_evaluate($dp, $LM, $cfg)
   Bgroup["grad_params_config"][ord, deg] = @benchmarkable ACE.grad_params_config($LM, $cfg)
   dq = [ACE.DState(rr=rand(SVector{3, Float64})) for _ = 1:length(cfg)] #the adjoint
   Bgroup["adjoint_EVAL_D"][ord, deg] = @benchmarkable ACE.adjoint_EVAL_D($LM, $LM.evaluator, $cfg, $dq)



   # #Zygote calls for derivatives
   # Bgroup["site_energy"][ord, deg] = @benchmarkable site_energy($LM, $cfg)
   # Bgroup["Zygote_grad_params"][ord, deg] = @benchmarkable gradient(x->site_energy(x,$cfg), $LM)
   # Bgroup["Zygote_grad_config"][ord, deg] = @benchmarkable gradient(x->site_energy($LM,x), $cfg)
   # #sum over the function so we don't compute the jacobian
   # Bgroup["Zygote_grad_params_config"][ord, deg] = @benchmarkable gradient(m->sum(sum(gradient(x->site_energy(m,x), $cfg)[1]).rr), $LM)



   # #loss function 
   # Bgroup["eval_full_loss"][ord, deg] = @benchmarkable loss($LM, $cfg)
   # Bgroup["eval_energy_loss"][ord, deg] = @benchmarkable lossE($LM, $cfg)
   # Bgroup["der_full_loss"][ord, deg] = @benchmarkable gradient(m->loss(m,$cfg), $LM)
   # Bgroup["der_energy_loss"][ord, deg] = @benchmarkable gradient(m->lossE(m,$cfg), $LM)


end 

##

linear_suite = BenchmarkGroup() 
linear_suite["P"] = Pgroup
linear_suite["B"] = Bgroup 

##

# @info("Tune")
# tune!(linear_suite)

# @info("Run")
# results = run(linear_suite, verbose = true)   
