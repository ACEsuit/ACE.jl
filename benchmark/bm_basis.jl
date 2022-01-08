
using ACE, Printf, BenchmarkTools

using ACE: evaluate, evaluate!, evaluate_d, evaluate_d!
using ACEbase: acquire_B!, acquire_dB!

TX = ACE.PositionState{Float64}
B1p = ACE.Utils.RnYlm_1pbasis()
Rn = B1p.bases[1]
cfg = ACEConfig(rand(TX, Rn, 30))

##

# degrees = Dict(2 => [9, 17],
#                3 => [7, 15] )

Adegrees = [10, 17]

degrees = Dict(2 => [10, 17],
               3 => [9, 15],
               5 => [7, 11] )
wL = 1.5 

##

Agroup = BenchmarkGroup()
Agroup["evaluate"] = BenchmarkGroup()
Agroup["evaluate_d"] = BenchmarkGroup()
Agroup["evaluate!"] = BenchmarkGroup()
Agroup["evaluate_d!"] = BenchmarkGroup()

for deg in Adegrees 
   local B1p
   Bsel = SparseBasis(; maxorder = 1, p = 1, default_maxdeg = deg,
                        weight = Dict(:n => 1.0, :l => wL))   
   B1p = ACE.Utils.RnYlm_1pbasis(maxdeg = deg, 
                                 maxL = ceil(Int, deg / wL), 
                                 Bsel = Bsel)
   A = evaluate(B1p, cfg)
   dA = evaluate_d(B1p, cfg)

   Agroup["evaluate"][deg] = @benchmarkable evaluate($B1p, $cfg)   
   Agroup["evaluate_d"][deg] = @benchmarkable evaluate_d($B1p, $cfg)   
   Agroup["evaluate!"][deg] = @benchmarkable evaluate!($A, $B1p, $cfg)   
   Agroup["evaluate_d!"][deg] = @benchmarkable evaluate_d!($dA, $B1p, $cfg)
end 

##

Bgroup = BenchmarkGroup()
Bgroup["evaluate"] = BenchmarkGroup()
Bgroup["evaluate_d"] = BenchmarkGroup()
Bgroup["evaluate!"] = BenchmarkGroup()
Bgroup["evaluate_d!"] = BenchmarkGroup()

for ord = keys(degrees), deg in degrees[ord]
   local B1p
   Bsel = SparseBasis(; maxorder = 1, p = 1, default_maxdeg = deg,
                        weight = Dict(:n => 1.0, :l => wL))   
   B1p = ACE.Utils.RnYlm_1pbasis(maxdeg = deg, 
                                 maxL = ceil(Int, deg / wL), 
                                 Bsel = Bsel)
   basis = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
   B = acquire_B!(basis, cfg)
   dB = acquire_dB!(basis, cfg)

   Bgroup["evaluate"][ord, deg] = @benchmarkable evaluate($basis, $cfg)   
   Bgroup["evaluate_d"][ord, deg] = @benchmarkable evaluate_d($basis, $cfg)   
   Bgroup["evaluate!"][ord, deg] = @benchmarkable evaluate!($B, $basis, $cfg)   
   Bgroup["evaluate_d!"][ord, deg] = @benchmarkable evaluate_d!($dB, $basis, $cfg)
end

##

basis_suite = BenchmarkGroup() 
basis_suite["A"] = Agroup
basis_suite["B"] = Bgroup 

##

# @info("Tune")
# tune!(basis_suite)

# @info("Run")
# results = run(basis_suite, verbose = true)   


