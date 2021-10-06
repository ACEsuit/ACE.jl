using ACE, Test, LinearAlgebra
using ACE: State, CylindricalBondEnvelope, ElipsoidBondEnvelope
using StaticArrays
using ACEbase.Testing: print_tf

## Use a specific example to test the Cylindrical cutoff

@info("Testing Cylindrical Bond Envelope")

r0cut = 8.0
rcut = 4.0
zcut = 2.0
env = CylindricalBondEnvelope(r0cut, rcut, zcut)

@info("Test :bond")

for i = 1:30
    rr = rand(SVector{3, Float64}) * 2 * r0cut / sqrt(3)
    r = norm(rr)
    X = State(rr = rr, rr0 = rr, be=:bond)
    print_tf(@test( filter(env, X) == (r <= r0cut) ))
end

##


@info ("Test :env")

r0 = SA[r0cut/2, 0.0, 0.0]
r_centre = r0 / 2
for i = 1:30
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


#%%

@info("Testing Elipsoid Bond Envelope")


r0cut = 2.0
rcut = 1.0
zcut = 2.0
for lambda = [0,.5,.6,1]
    for floppy = [false, true]
        env = ElipsoidBondEnvelope(r0cut, rcut, zcut;floppy=floppy, λ= lambda)

        @info("Test :bond", floppy, lambda)

        for i = 1:30
            rr = rand(SVector{3, Float64}) * 2 * r0cut / sqrt(3)
            r = norm(rr)
            X = State(rr = rr, rr0 = rr, be=:bond)
            print_tf(@test( filter(env, X) == (r <= r0cut) ))
        end

        ##


        @info ("Test :env")

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
    end
end


#%%

@info("Testing a simple bond basis")


const bsymbols = (:bond,:env)

#onst ExtendedAtomState{T} = ACE.State{(:rr, :be), Tuple{SVector{3, T}, Symbol}}

#ExtendedAtomState(as::AtomState{T}, s::Symbol) = begin
#   @assert s in bsymbols
#   ExtendedAtomState{T}( (:be = s, rr = as.rr) )
#end

using ACE: State, ElipsoidBondEnvelope, CategoryBasisSelector

function Bond_basis(; init = true,
                           maxL = 4,
                           maxdeg = 6,
                           Bsel = nothing,
                           maxorder = 2,
                           kwargs...)
    if Bsel == nothing
        maxorder_dict = Dict(:bond => 1)
        isym = :be
        weight = Dict(:l =>1, :n => 1)
        degree = Dict("default" => 10)
        p = 2
        Bsel = CategoryBasisSelector(maxorder, maxorder_dict, isym, weight, degree, p)
    end
    RnYlm = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, maxL=maxL, kwargs...)
    Aν = Categorical1pBasis([:env,:bond]; varsym = :be, idxsym = :be)
    B1p = Aν * RnYlm
    basis = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
    if init
        ACE.init1pspec!(B1p, Bsel)
    end
    return basis
end


basis = Bond_basis(; init = true, Bsel = nothing, maxorder = 2)


using ACE

# CategoricalBasis
categories = [:e, :b]
len = length(categories)
list = ACE.SList(categories)
B1p_be = ACE.Categorical1pBasis(categories; varsym = :be, idxsym=:be)

# RnYlmBasis
maxdeg = 6
ord = 2
Bsel = SimpleSparseBasis(ord, maxdeg)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel=Bsel)

# Combined
B1p_test = B1p_be * B1p
basis =  ACE.init1pspec!(B1p_test,Bsel)

ACE.SymmetricBasis(ACE.Invariant(), basis, Bsel)
