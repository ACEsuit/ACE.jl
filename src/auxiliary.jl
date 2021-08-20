

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




# the following two functions are a little hack to make sure 
# that the basis spec is read in the same symbol-order as it is written
# (since dicts don't have a specified ordering...)

function _write_dict_1pspec(spec::Vector{NamedTuple{SYMS, NTuple{NSYM, Int}}}) where {SYMS, NSYM}
   inds = Vector{Int}[]
   for b in spec 
      vals = [getproperty(b, sym) for sym in SYMS]
      push!(inds, vals)
   end 
   return Dict("SYMS" => [ string.(SYMS)... ], "inds" => inds)
end

function _read_dict_1pspec(D::Dict)
   NTPROTO = namedtuple(D["SYMS"]...)
   return [ NTPROTO(binds) for binds in D["inds"] ]
end
