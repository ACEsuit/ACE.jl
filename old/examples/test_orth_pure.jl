
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------

using Combinatorics

VecTup = Union{AbstractVector, Tuple}




"""
This implements
```
   (N!)^{-1} ∑_{σ1, σ2} < Y_l1^m1 ∘ σ1, Y_l2^m2 ∘ σ2 >
```
"""
symdotYlm(l1, m1, l2, m2) =
   l1 != l2 ? 0 :  sum( (l1[σ] == l2) * (m1[σ] == m2)
                        for σ in permutations(1:length(l1)) )


function randlm(N, maxl=4)
   l = sort( rand(0:maxl, N) )
   m1 = [ rand(-l[a]:l[a]) for a = 1:N ]
   m2 = [ rand(-l[a]:l[a]) for a = 1:N ]
   return tuple(l...), tuple(m1...), tuple(m2...)
end

l, m1, m2 = randlm(6)

using BenchmarkTools
@btime symdotYlm($l, $m1, $l, $m2)

symdotYlm(l, m1, l, m2)
