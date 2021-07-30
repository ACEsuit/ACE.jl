

"""
random vector on unit sphere, should be uniformly distributed?
"""
function rand_sphere(T = Float64)
   R = randn(SVector{3, T})
   return R / norm(R)
end

"""
random rotation matrix; 
WARN: never checked what the distribution is
"""
rand_rot() = (K = (@SMatrix rand(3,3)) .- 0.5; exp(K - K'))

"""
random reflection, represented as 1, -1 Integer
"""
rand_refl() = rand([-1,1])

"""
random isometry i.e. element of O(3)
"""
rand_O3() = rand_refl() * rand_rot()
