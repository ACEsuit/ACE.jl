
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using JuLIP, SHIPs, Test, IPFitting, DataFrames
using IPFitting: Dat
using JuLIP.MLIPs

# generate random data
function generate_data(species, L, rmax, N, calc)
   data = Dat[]
   for n = 1:N
      at = bulk(species; cubic=true, pbc=true) * L
      rattle!(at, rand() * rmax)
      E = energy(calc, at)
      F = forces(calc, at)
      push!(data, Dat(at, "rand"; E = E, F = F))
   end
   return data
end

r0 = rnn(:Si)
calc = StillingerWeber()
train = generate_data(:Si, 2, 0.2*r0, 1000, calc)


# 2 stands for 2 neighbours i.e. body-order 3
basis(deg) = IPSuperBasis(
      PairBasis(deg, PolyTransform(2, r0), 2, cutoff(calc)),
      SHIPBasis(2, deg, 2.0, PolyTransform(3, r0), 2, 0.5*r0, cutoff(calc))
   )

##
err_erms = Float64[]
err_frms = Float64[]
degrees = [4, 8, 12, 16, 20]

for deg in degrees
   shipB = basis(deg)
   @show length(shipB)
   err = SHIPs.Lsq.lsqfit(train, shipB;
                           configweights = Dict("rand" => 1.0),
                           obsweights   = Dict("E" => 1.0, "F" => 1.0),
                           verbose=true)
   @show err
end


# ##
# df = DataFrame( :degrees => degrees,
#                 :relrms_E => err_erms,
#                 :relrms_F => err_frms )
# display(df)
#
# (@test minimum(err_erms) < 0.001) |> println
# (@test minimum(err_frms) < 0.05) |> println
#
