

import ACE
using LinearAlgebra, StaticArrays, BenchmarkTools, Test, Printf
using ACE: evaluate, evaluate_d!, evaluate_ed, alloc_B, release!, 
           evaluate!, acquire!, acquire_B!, acquire_dB!, 
           release_B!, release_dB!, evaluate_d
using ACE.Transforms: PolyTransform 
using ACE.Testing
using BenchmarkTools

##

trans = PolyTransform(2, 1.0)
P = ACE.OrthPolys.transformed_jacobi(20, trans, 2.0, 0.5; pcut = 2, pin = 2)

B = acquire_B!(P)
dB = acquire_dB!(P)
r = ACE.rand_radial(P)

@info("non-allocating (for reference)")
@btime evaluate!($B, $P, $r)
@btime evaluate_d!($dB, $P, $r)

@info("fully allocating")
@btime evaluate($P, $r)
@btime evaluate_d($P, $r)

@info("allocate and release")
@info("In this case there seems to be limited advantage?")
_ev(P, r) = (B = evaluate!(acquire_B!(P, r), P, r); release_B!(P, B))
_ev_d(P, r) = (dB = evaluate_d!(acquire_dB!(P, r), P, r); release_dB!(P, dB))
@btime _ev($P, $r)
@btime _ev_d($P, $r)
