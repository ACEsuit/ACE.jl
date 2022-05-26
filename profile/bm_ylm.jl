

import ACE
using LinearAlgebra, StaticArrays, BenchmarkTools, Test, Printf
using ACE.SphericalHarmonics
using ACE.SphericalHarmonics: dspher_to_dcart, SphericalCoords,
               cart2spher, spher2cart, ALPolynomials
using ACE: evaluate, evaluate_d, evaluate_ed, alloc_B, release!, 
           evaluate!, acquire!, acquire_B!, acquire_dB!
using ACE.Testing
using BenchmarkTools

##

L = 10 

alp = ALPolynomials(L, Float64)


S = SphericalCoords(0.0, 0.0, rand() * π)

const pl = ACE.ObjectPools.ArrayPool()
const hpl = ACE.ObjectPools.HomogeneousVectorPool{Float64}()
const stpl = ACE.ObjectPools.StaticVectorPool{Float64}()


function _ev(b, x)
   B = acquire!(pl, Float64, (length(b),))
   evaluate!(B, b, x)
   release!(pl, B)
end

function _hev(b, x)
   B = acquire!(hpl, length(b))
   evaluate!(B, b, x)
   release!(hpl, B)
end

function _stev(b, x)
   B = acquire!(stpl, length(b))
   evaluate!(B, b, x)
   release!(stpl, B)
end

B = ACE.acquire_B!(alp)

##

@info("ALP")
@info("non-allocating (for reference)")
@btime evaluate!($B, $alp, $S)
@info("ArrayPool")
@btime _ev($alp, $S)
@info("HomogeneousVectorPool")
@btime _hev($alp, $S)
@info("StaticVectorPool")
@btime _stev($alp, $S)


##

@info("Spherical Harmonics")
sh = SHBasis(L)
x = randn(SVector{3, Float64})
B_sh = acquire_B!(sh)
P_sh = acquire_B!(sh.alp)

@info("non-allocating")
@btime ACE.SphericalHarmonics.__evaluate!($B_sh, $sh, $P_sh, $x)
@info("ALP allocates via pool, Bsh is passed in")
@btime evaluate!($B_sh, $sh, $x)

##

@info("Spherical Harmonics gradients")
dB_sh = ACE.SphericalHarmonics.acquire_dB!(sh, x)
dP_sh = acquire_dB!(sh.alp)

@info("non-allocating")
@btime ACE.SphericalHarmonics.__evaluate_ed!($B_sh, $dB_sh, $sh, $P_sh, $dP_sh, $x)
@info("ALP allocates via pool, Bsh is passed in")
@btime ACE.evaluate_ed!($B_sh, $dB_sh, $sh, $x)
