module Wigner

using StaticArrays
#using ACE: SphericalVector
#import ACE.Rotations3D.Rotation_D_matrix
#import ACE. getL

#export rot_D

# Index of entries in D matrix (sign free)
struct D_Index
	l::Int64
	μ::Int64
	m::Int64
end

# Equation (1.1) - forms the covariant matrix D(Q)(indices only)
function Rotation_D_matrix(L::Integer)
	if L<0
		error("Orbital type shall be represented as a positive integer!")
	end
    D = Array{D_Index}(undef, 2 * L + 1, 2 * L + 1)
    for i = 1 : 2 * L + 1
        for j = 1 : 2 * L + 1
            D[j,i] = D_Index(L, i - 1 - L, j - 1 - L);
        end
    end
	return D
end

function Rotation_D_matrix_ast(L::Integer)
	if L<0
		error("Orbital type shall be represented as a positive integer!")
	end
    D = Array{D_Index}(undef, 2 * L + 1, 2 * L + 1)
    for i = 1 : 2 * L + 1
        for j = 1 : 2 * L + 1
            D[i,j] = D_Index(L, -(i - 1 - L), -(j - 1 - L));
        end
    end
	return D
end


function Wigner_D(μ,m,l,α,β,γ)
	return (exp(-im*α*m) * wigner_d(m,μ,l,β)  * exp(-im*γ*μ))'
end

# Wigner small d, modified from
# https://github.com/cortner/SlaterKoster.jl/blob/
# 8dceecb073709e6448a7a219ed9d3a010fa06724/src/code_generation.jl#L73
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

function Mat2Ang(Q)
	return mod(atan(Q[2,3],Q[1,3]),2pi), acos(Q[3,3]), mod(atan(Q[3,2],-Q[3,1]),2pi);
end

# Rotation D matrix
function rot_D(L::Integer, Q)
	Mat_D = zeros(ComplexF64, 2L + 1, 2L + 1);
	D = Rotation_D_matrix(L);
	α, β, γ = Mat2Ang(Q);
	for i = 1 : 2L + 1
		for j = 1 : 2L + 1
			Mat_D[i,j] = Wigner_D(D[i,j].μ, D[i,j].m, D[i,j].l, α, β, γ);
		end
	end
	return SMatrix{2L+1, 2L+1, ComplexF64}(Mat_D)
end



function rand_QD(L)
	rotz(α) = [cos(α) -sin(α) 0; sin(α) cos(α) 0; 0 0 1]
	roty(α) = [cos(α) 0 sin(α); 0 1 0;-sin(α) 0 cos(α)]
	Ang2Mat_zyz(α,β,γ) = rotz(α)*roty(β)*rotz(γ)

	α = 2pi*rand();
	β = pi*rand();
	γ = 2pi*rand();

	# construct the Q matrix
	Q = Ang2Mat_zyz(α,β,γ)
	Q = SMatrix{3,3}(Q)

	return Q, rot_D(L, Q)
end

end
