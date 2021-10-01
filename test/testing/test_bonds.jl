using ACE, Test
using ACE: State, CylindricalBondEnvelope

## Use a specific example to test the Cylindrical cutoff

@info("Testing Cylindrical Bond Envelope")

Xs1 = State(rr = [6,0,0], rr0 = [6,0,0], be=:bond)
Xs2 = State(rr = [9,0,0], rr0 = [9,0,0], be=:bond)
Xs3 = State(rr = [7,0,2], rr0 = [6,0,0], be=:env)
Xs4 = State(rr = [9,0,2], rr0 = [6,0,0], be=:env)

env = CylindricalBondEnvelope(8.0,4.0,2.0)

# TODO: using random states (w.r.t. `env`) to replace these specific ones

println(@test( filter(env,Xs1) == true ))
println(@test( filter(env,Xs2) == false ))
println(@test( filter(env,Xs3) == true ))
println(@test( filter(env,Xs4) == false ))
