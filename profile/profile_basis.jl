
using ACE, Printf, BenchmarkTools, 
      Profile, ProfileSVG

using ACE: evaluate, evaluate!, evaluate_d, evaluate_d! 
using ACEbase: acquire_B!, acquire_dB!

TX = ACE.PositionState{Float64}
B1p = ACE.Utils.RnYlm_1pbasis()
Rn = B1p.bases[1]
cfg = ACEConfig(rand(TX, Rn, 30))

##


# degrees = Dict(2 => [7, 12, 17],
#                3 => [7, 11, 15],
#                4 => [7, 10, 13], 
#                5 => [7, 9, 11] )

# bmgroup = BenchmarkGroup()
# bmgroup["evaluate"] = BenchmarkGroup()
# bmgroup["evaluate_d"] = BenchmarkGroup()
# bmgroup["evaluate!"] = BenchmarkGroup()
# bmgroup["evaluate_d!"] = BenchmarkGroup()

# for ord = 2:3, deg in degrees[ord]
#    Bsel = ACE.SimpleSparseBasis(ord, deg)
#    B1p = ACE.Utils.RnYlm_1pbasis(maxdeg = deg, Bsel = Bsel)
#    basis = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
#    B = acquire_B!(basis, cfg)
#    dB = acquire_dB!(basis, cfg)

#    bmgroup["evaluate"][ord, deg] = @benchmarkable evaluate($basis, $cfg)   
#    bmgroup["evaluate_d"][ord, deg] = @benchmarkable evaluate($basis, $cfg)   
#    bmgroup["evaluate!"][ord, deg] = @benchmarkable evaluate($basis, $cfg)   
#    bmgroup["evaluate_d!"][ord, deg] = @benchmarkable evaluate($basis, $cfg)   
# end

##

# tune!(bmgroup)

# results = run(bmgroup, verbose = true)

# ##

# plot(results["evaluate"])


## Manual Profiling Codes

ord = 3
deg = 15
Bsel = ACE.SimpleSparseBasis(ord, deg)

B1p = ACE.Utils.RnYlm_1pbasis(maxdeg = deg, Bsel = Bsel)
Rn = B1p.bases[1]
Ylm = B1p.bases[2]
basis = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)

TX = ACE.PositionState{Float64}
cfg = ACEConfig(rand(TX, Rn, 50))
B1 = acquire_B!(B1p, cfg)

##

# the non-allocating version here makes virtually no difference
# Further, the move from Set to Stack fixed all the weird performance 
# problems; it seems our 1p basis evaluation is now very fast. 

@info("profile A")
A = acquire_B!(B1p, cfg)
@btime evaluate($B1p, $cfg)
@btime evaluate!($A, $B1p, $cfg)

##

@info("profile dA")
dA = evaluate_d(B1p, cfg)
@btime evaluate_d($B1p, $cfg)
@btime evaluate_d!($dA, $B1p, $cfg)

##

# @info("profile B")
# B = acquire_B!(basis, cfg)
# @btime evaluate($basis, $cfg)
# @btime evaluate!($B, $basis, $cfg)

##

# @info("profile AA")
# AA = acquire_B!(basis.pibasis, cfg)
# @btime evaluate!($AA, $(basis.pibasis), $cfg)

##

# dB = acquire_dB!(basis, cfg)
# @btime evaluate_d($basis, $cfg)
# @btime evaluate_d!($dB, $basis, $cfg)

##


function runn(N, f, args...)
   for n = 1:N 
      r = f(args...)
   end
end

runn(2, evaluate_d, B1p, cfg)

##
Profile.clear()
@profile runn(100, evaluate_d, B1p, cfg)
Profile.print()
