using ACE, Test, LinearAlgebra
using ACE: State, CylindricalBondEnvelope

## Use a specific example to test the Cylindrical cutoff

@info("Testing Cylindrical Bond Envelope")

r0cut = 8.0
rcut = 4.0
zcut = 2.0
env = CylindricalBondEnvelope(r0cut,rcut,zcut)

@info("Test :bond")

for i = 1:30
    r = rand(Float64,3)*2*r0cut/sqrt(3)
    Xs = State(rr = r, rr0 = r, be=:bond)
    println(@test( filter(env,Xs) == (norm(r)<=8) ))
end

@info ("Test :env")

r0 = [r0cut/2,0.0,0.0]
r_centre = r0/2
for i = 1:30
    r = rand(Float64,3)*env.rcut + r_centre
    Xs = State(rr = r, rr0 = r0, be=:env)
    println(@test( filter(env,Xs) == ((abs(r[1]-r_centre[1])≤env.zcut+r_centre[1]) && (r[2]^2+r[3]^2≤env.rcut^2)) ))
end
