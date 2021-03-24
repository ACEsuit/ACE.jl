

using StaticArrays, ACE, BenchmarkTools, LinearAlgebra
using ACE.SphericalHarmonics
SH = ACE.SphericalHarmonics
using ACE: alloc_temp, alloc_temp_d, alloc_B, alloc_dB, evaluate!, evaluate_d!

@info("Spherical Harmonics Evaluation")

##

suite = BenchmarkGroup()

# (@benchmark runmany!($Y, $tmp, $basis, 100)) |> display
# @btime runmany!($Y, $tmp, $basis, 100)
#
# R = ACE.rand_sphere()
# @btime evaluate!($Y, $tmp, $basis, $R)

R = JVecF(0.4, 0.7, -0.9)

for L in [5, 10, 15]
   basis = SH.SHBasis(L)
   tmp = alloc_temp(basis)
   Y = alloc_B(basis)
   tmp_d = alloc_temp_d(basis)
   dY = alloc_dB(basis)
   suite["Complex SH($L) - evaluate!"] =
         (@benchmarkable evaluate!($Y, $tmp, $basis, $R))
   suite["Complex SH($L) - evaluate_d!"] =
         (@benchmarkable evaluate_d!($Y, $dY, $tmp_d, $basis, $R))

   rbasis = SH.RSHBasis(L)
   rtmp = alloc_temp(rbasis)
   rY = alloc_B(rbasis)
   rtmp_d = alloc_temp_d(rbasis)
   rdY = alloc_dB(rbasis)
   suite["Real SH($L) - evaluate!"] =
         (@benchmarkable evaluate!($rY, $rtmp, $rbasis, $R))
   suite["Real SH($L) - evaluate_d!"] =
         (@benchmarkable evaluate_d!($rY, $rdY, $rtmp_d, $rbasis, $R))
end


if !isdefined(Main, :globalsuite)
   tune!(suite)
   results = run(suite; verbose=true)
   display(results)
else
   globalsuite["ylm"] = suite
end
