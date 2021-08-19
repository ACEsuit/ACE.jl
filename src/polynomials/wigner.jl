
# this module is currently only used for setup and testing covariance properties
# the spherically covariant ACE basis. There are few/no optimisations for
# performance.
module Wigner

using StaticArrays
using ACE: rand_rot, rand_refl

__L2syms = [:s, :p, :d, :f, :g, :h, :i, :k]
__syms2L = Dict( [sym => L-1 for (L, sym) in enumerate(__L2syms)]... )
get_orbsym(L::Integer)  = __L2syms[L+1]


"""
Index of entries in D matrix (sign included)
"""
struct D_Index
	sign::Int64
	μ::Int64
	m::Int64
end

"""
auxiliary matrix - indices for D matrix
"""
wigner_D_indices(L::Integer) = (   @assert L >= 0;
		[ D_Index(1, i - 1 - L, j - 1 - L) for j = 1:2*L+1, i = 1:2*L+1] )

Base.adjoint(idx::D_Index) = D_Index( (-1)^(idx.μ+idx.m), - idx.μ, - idx.m)

"""
One entry of the Wigner-big-D matrix, `[D^l]_{mu, m}`
"""
wigner_D(μ,m,l,α,β,γ) = (exp(-im*α*m) * wigner_d(m,μ,l,β)  * exp(-im*γ*μ))'



"""
One entry of the Wigner-small-d matrix,

Wigner small d, modified from
```
https://github.com/cortner/SlaterKoster.jl/blob/
8dceecb073709e6448a7a219ed9d3a010fa06724/src/code_generation.jl#L73
```
"""
function wigner_d(μ, m, l, β)
    fc1 = factorial(l+m)
    fc2 = factorial(l-m)
    fc3 = factorial(l+μ)
    fc4 = factorial(l-μ)
    fcm1 = sqrt(fc1 * fc2 * fc3 * fc4)

    cosb = cos(β / 2.0)
    sinb = sin(β / 2.0)

    p = m - μ
    low  = max(0,p)
    high = min(l+m,l-μ)

    temp = 0.0
    for s = low:high
       fc5 = factorial(s)
       fc6 = factorial(l+m-s)
       fc7 = factorial(l-μ-s)
       fc8 = factorial(s-p)
       fcm2 = fc5 * fc6 * fc7 * fc8
       pow1 = 2 * l - 2 * s + p
       pow2 = 2 * s - p
       temp += (-1)^(s+p) * cosb^pow1 * sinb^pow2 / fcm2
    end
    temp *= fcm1

    return temp
end

mat2ang(Q) = mod(atan(Q[2,3],Q[1,3]),2pi), acos(Q[3,3]), mod(atan(Q[3,2],-Q[3,1]),2pi);


function wigner_D(L::Integer, Q::AbstractMatrix)
	D = wigner_D_indices(L);
	α, β, γ = mat2ang(Q);
	Mat_D = [ wigner_D(D[i,j].μ, D[i,j].m, L, α, β, γ)
			    for i = 1:2*L+1, j = 1:2*L+1 ]
   # NB: type instability here, but performance is not important.
	return SMatrix{2L+1, 2L+1, ComplexF64}(Mat_D)
end


function rand_QD(L::Integer)
	# random rotation matrix, reflection
	Q, σ = rand_rot(), rand_refl()
	# return resulting D matrix
	return σ * Q, σ^L * wigner_D(L, Q)
end

function rand_QD(L1::Integer, L2::Integer)
	# random rotation matrix, reflection
	Q, σ = rand_rot(), rand_refl()
	# return Q, D_L1, D_L2
	return σ * Q, σ^(L1) * wigner_D(L1, Q), σ^(L2) * wigner_D(L2, Q)
end

end
