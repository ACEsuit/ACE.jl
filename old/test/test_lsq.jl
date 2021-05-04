
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using JuLIP, ACE, Test, IPFitting, DataFrames
using IPFitting: Dat
using JuLIP.MLIPs
using Printf

# generate random data
function generate_data(species, L, rmax, N, calc)
   data = Dat[]
   for n = 1:N
      at = bulk(species; cubic=true, pbc=true) * L
      rattle!(at, rand() * rmax)
      E = energy(calc, at)
      F = forces(calc, at)
      V = virial(calc, at)
      push!(data, Dat(at, "rand"; E = E, F = F, V = V))
   end
   return data
end

r0 = rnn(:Si)
calc = StillingerWeber()
train = generate_data(:Si, 2, 0.2*r0, 300, calc)


# 2 stands for 2 neighbours i.e. body-order 3
basis(deg) = IPSuperBasis(
      PairBasis(deg, PolyTransform(2, r0), 2, cutoff(calc)),
      SHIPBasis(SparseSHIP(deg, 2.0), 2, rbasis(PolyTransform(3, r0), 2, 0.5*r0, cutoff(calc)))
   )

# basis(deg) = SHIPBasis(2, deg, 2.0, PolyTransform(3, r0), 2, 0.5*r0, cutoff(calc))


##
err_rms = Float64[]
degrees = [4, 8, 12, 16, 20]

@printf(" degree | #basis  RMSE \n")
for deg in degrees
   shipB = basis(deg)
   err = ACE.Lsq.lsqfit(train, shipB;
                          configweights = Dict("rand" => 1.0),
                          obsweights   = Dict("E" => 1.0, "F" => 1.0, "V" => 1.0),
                          verbose=false)
   @printf("    %2d  |  %4d   %.2e \n", deg, length(shipB), err)
   push!(err_rms, err)
end
