using ACE, Test, LinearAlgebra
using ACE: State, CylindricalBondEnvelope, EllipsoidBondEnvelope
using StaticArrays 
using ACEbase.Testing: print_tf

using ACE.Random: rand_rot, rand_refl
## Use a specific example to test the Cylindrical cutoff

@info("Testing Cylindrical Bond Envelope")

r0cut = 8.0
rcut = 4.0
zcut = 2.0
env = CylindricalBondEnvelope(r0cut, rcut, zcut)

@info("Test :bond")

for i = 1:30
    local X
    rr = rand(SVector{3, Float64}) * 2 * r0cut / sqrt(3)
    r = norm(rr) 
    X = State(rr = rr, rr0 = rr, be=:bond)
    print_tf(@test( filter(env, X) == (r <= r0cut) ))
end
println()

##


@info ("Test :env")

r0 = SA[r0cut/2, 0.0, 0.0]
r_centre = r0 / 2
for i = 1:30
    local X 
    rr = rand(SVector{3, Float64}) * env.rcut + r_centre
    X = State(rr = rr, rr0 = r0, be=:env)
    z = rr[1] - r_centre[1]
    r = sqrt(rr[2]^2 + rr[3]^2)
    filt = (abs(z) <= env.zcut+r_centre[1]) * (r <= env.rcut) 
    print_tf(@test( filter(env, X) == (filt != 0) ))

    zeff = env.zcut + norm(r_centre)
    val = ((r/env.rcut)^2 - 1)^env.pr * ( (z/zeff)^2 - 1 )^env.pz * filt
    print_tf(@test( ACE._inner_evaluate(env, X) ≈ val ))
end
println()

#%%

@info("Testing Ellipsoid Bond Envelope")


r0cut = 2.0
rcut = 1.0
zcut = 2.0
for lambda = [0,.5,.6,1]
    for floppy = [false, true]
        local env
        env = EllipsoidBondEnvelope(r0cut, rcut, zcut;floppy=floppy, λ= lambda)

        @info("Test :bond", floppy, lambda)

        for i = 1:30
            local X 
            rr = rand(SVector{3, Float64}) * 2 * r0cut / sqrt(3)
            r = norm(rr)
            X = State(rr = rr, rr0 = rr, be=:bond)
            print_tf(@test( filter(env, X) == (r <= r0cut) ))
        end
        println()

        ##


        @info ("Test :env")
        local r0, r_centre, X 
        r0 = @SVector [r0cut/2, 0.0, 0.0]
        r_centre = r0 * env.λ
        for i = 1:30
            rr = rand(SVector{3, Float64}) * env.rcut + r_centre
            X = State(rr = rr, rr0 = r0, be=:env)
            z = rr[1] - r_centre[1]
            r = sqrt(rr[2]^2 + rr[3]^2)
            zeff = env.zcut + env.floppy*norm(r_centre)

            filt = (((z/ zeff)^2 + (r/env.rcut)^2) <=1)

            #@show filt
            #@show filter(env, X)
            #print("------------------- \n")
            print_tf(@test( filter(env, X) == (filt != 0) ))

            val = ( (z/zeff)^2 +  (r/env.rcut)^2 - 1.0)^env.pr * filt
            #@show val
            #@show ACE._inner_evaluate(env, X)
            #print("------------------- \n")
            print_tf(@test( ACE._inner_evaluate(env, X) ≈ val ))
        end
        println()
    end
end
