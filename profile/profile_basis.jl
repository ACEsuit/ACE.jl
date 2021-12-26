using ACE, Printf, BenchmarkTools

using ACE: evaluate, evaluate!, evaluate_d, evaluate_d! 
using ACEbase: acquire_B!, acquire_dB!

TX = ACE.PositionState{Float64}
B1p = ACE.Utils.RnYlm_1pbasis()
Rn = B1p.bases[1]
cfg = ACEConfig(rand(TX, Rn, 30))


## Manual Profiling Codes

ord = 3
deg = 15
wL = 1.5 
# Bsel = ACE.SimpleSparseBasis(ord, deg)
Bsel = SparseBasis(; maxorder = 3, p = 1, default_maxdeg = deg, 
                     weight = Dict(:n => 1.0, :l => wL))

B1p = ACE.Utils.RnYlm_1pbasis(maxdeg = deg, maxL = ceil(Int, deg/wL), 
                              Bsel = Bsel)
Rn = B1p.bases[1]
Ylm = B1p.bases[2]
basis = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)

TX = ACE.PositionState{Float64}
cfg = ACEConfig(rand(TX, Rn, 20))
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

@info("profile B")
B = acquire_B!(basis, cfg)
@btime evaluate($basis, $cfg)
@btime evaluate!($B, $basis, $cfg)

##

@info("profile AA")
AA = acquire_B!(basis.pibasis, cfg)
@btime evaluate!($AA, $(basis.pibasis), $cfg)

##

@info("profile dB")
dB = acquire_dB!(basis, cfg)
@btime evaluate_d($basis, $cfg)
@btime evaluate_d!($dB, $basis, $cfg)

# this fits with cost(dB) ≈ cost(B) * length(cfg) * 2 
# so it seems close enough to optimal to accept this. 

## ------- profiling codes 

using Profile, ProfileSVG

function runn(N, f, args...)
   for n = 1:N 
      r = f(args...)
   end
end

B = evaluate(basis, cfg)
runn(100, evaluate!, B, basis, cfg)

##

@code_warntype evaluate!(B, basis, cfg)
@code_warntype acquire_B!(basis.pibasis, cfg)

@code_warntype ACE.valtype(basis.pibasis)

# ##

# fill!(B, zero(eltype(B)))

# Profile.clear()
# @profile runn(10_000, evaluate!, B, basis, cfg)
# Profile.print()

# ##

# ProfileSVG.view()

