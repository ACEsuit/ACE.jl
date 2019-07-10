
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@info("Load libraries...")
using JuLIP, SHIPs, IPFitting
using JuLIP.MLIPs: IPSuperBasis
Err = IPFitting.Errors

include("benchmarks.jl")
using Main.ShipBenchmarks: append_benchmark!, init_benchmark
bfile = "W_benchmarks2.json"
init_benchmark(bfile, false)


@info("Load W database...")
include("W.jl")
cfgs = W.loaddb("W.xyz")

# Itrain, Itest = IPFitting.splittraintest(cfgs, 0.5)
# cfgs = cfgs[Itrain]
@show length(cfgs)

@show E0 = -9.19483512529700

cfgweights = Dict(
   "gamma_surface_vacancy"  =>  1.0,
   "vacancy"  =>  1.0,
   "slice_sample"  =>  1.0,
   "gamma_surface"  =>  1.0,
   "surface"  =>  1.0,
   "md_bulk"  =>  1.0,
   "dislocation_quadrupole"  =>  1.0,
   )

obsweights = Dict( "E" => 0.1, "F" => 1.0, "V" => 0.01 )


r0 = round(rnn(:W), digits=2) # 2.74
rcut0 = 0.7 * rnn(:W)
rcut2 = 3 * rnn(:W)
rcut3 = 4.9 # 2.6 * rnn(:W)
rcut4 = 2.3 * rnn(:W)
rcut5 = 5.0 # 1.9 * rnn(:W)
rcut = 5.0 # SAME AS GAP

## First basis
B2 = PairBasis(20, PolyTransform(1, r0), 2, rcut2)
basis3(deg, wY) = SHIPBasis(TotalDegree(deg, wY), 2, PolyTransform(3, r0), 2, rcut0, rcut)
basis4(deg, wY) = SHIPBasis(TotalDegree(deg, wY), 3, PolyTransform(3, r0), 2, rcut0, rcut)
basis5(deg, wY) = SHIPBasis(TotalDegree(deg, wY), 4, PolyTransform(3, r0), 2, rcut0, rcut)


##

# 3B TEST
@info("3B Tests")
for wY in [1.0, 1.5, 2.0], deg in [12, 14, 16, 20, 24]
   shipB = IPSuperBasis(B2, basis3(deg,wY))
   if length(shipB) > 5000; continue; end
   @info("deg = $deg, wY = $wY, len(shipB) = $(length(shipB))")
   lsqsys = LsqDB("", shipB, cfgs)
   GC.gc()

   IP, lsqinfo = lsqfit(  lsqsys; E0 = E0,
                          obsweights = obsweights,
                          configweights = cfgweights,
                          solver = (:qr,)  )
   GC.gc()
   Err.rmse_table(Err.rmse(lsqinfo["errors"])...)

   append_benchmark!(IP, shipB, lsqinfo, :W, bfile)
end


# 4B TEST
@info("4B Test")
for wY in [1.0, 1.5, 2.0], deg in [10, 13, 16, 20, 24]
   shipB = IPSuperBasis( B2,
                         basis4(deg, wY) )
   if length(shipB) > 6500; continue; end
   @info("deg = $deg, wY = $wY, len(shipB) = $(length(shipB))")
   lsqsys = LsqDB("", shipB, cfgs)
   GC.gc()

   IP, lsqinfo = lsqfit(  lsqsys; E0 = E0,
                           obsweights = obsweights,
                           configweights = cfgweights,
                           solver = (:qr,)  )
   GC.gc()
   Err.rmse_table(Err.rmse(lsqinfo["errors"])...)
   append_benchmark!(IP, shipB, lsqinfo, :W, bfile)
end


# 5B TEST
@info("5B Test")
for wY in [1.0, ], deg in [9, 11, 12, 13, 14, 15]
   shipB = IPSuperBasis( B2,
                         basis5(deg, wY) )
   if length(shipB) > 6500; continue; end
   @info("deg = $deg, wY = $wY, len(shipB) = $(length(shipB))")
   lsqsys = LsqDB("", shipB, cfgs)
   GC.gc()

   IP, lsqinfo = lsqfit(  lsqsys; E0 = E0,
                           obsweights = obsweights,
                           configweights = cfgweights,
                           solver = (:qr,)  )
   lsqsys = nothing
   GC.gc()
   Err.rmse_table(Err.rmse(lsqinfo["errors"])...)
   append_benchmark!(IP, shipB, lsqinfo, :W, bfile)
end

# 5B TEST
@info("5B Test")
for wY in [1.5, ], deg in [8, 10, 12, 15, 16, 17]  # 18, 19]
   shipB = IPSuperBasis( B2,
                         basis5(deg, wY) )
   if length(shipB) > 6800; continue; end
   @info("deg = $deg, wY = $wY, len(shipB) = $(length(shipB))")
   lsqsys = LsqDB("", shipB, cfgs)
   GC.gc()

   IP, lsqinfo = lsqfit(  lsqsys; E0 = E0,
                           obsweights = obsweights,
                           configweights = cfgweights,
                           solver = (:qr,)  )
   lsqsys = nothing
   GC.gc()
   Err.rmse_table(Err.rmse(lsqinfo["errors"])...)
   append_benchmark!(IP, shipB, lsqinfo, :W, bfile)
end


# 5B TEST
@info("5B Test")
for wY in [2.0, ], deg in [12, 15, 17, 18, 19, 20]
   shipB = IPSuperBasis( B2,
                         basis5(deg, wY) )
   if length(shipB) > 6800; continue; end
   @info("deg = $deg, wY = $wY, len(shipB) = $(length(shipB))")
   lsqsys = LsqDB("", shipB, cfgs)
   GC.gc()

   IP, lsqinfo = lsqfit(  lsqsys; E0 = E0,
                           obsweights = obsweights,
                           configweights = cfgweights,
                           solver = (:qr,)  )
   lsqsys = nothing
   GC.gc()
   Err.rmse_table(Err.rmse(lsqinfo["errors"])...)
   append_benchmark!(IP, shipB, lsqinfo, :W, bfile)
end
