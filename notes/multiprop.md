
It is not entirely obvious how multiple property ought to be handled. 

`c = Vector{SVector{NP, T}}` where normally `T = Float64`. Then 
```
   p = sum_i ci Bi
```
Now, what is `∂p / ∂c`? In index-notation, it is quite clear; 
```
  ∂p_a / ∂ci_b = Bi * (a==b)
```
So, should `∂p / ∂c` be a three-dimensional tensor indexed by `i, a, b`? Seems fairly clear. Specifically, 
```
   `∂p / ∂c = [ ∂p / ∂c_i ]_i`
```
where each entry of this vector will be an `SMatrix`. This is of course extremely wasteful, but if we use a `SDiagonal` it is not too bad. Ideally of course it would be a type that stores this just once. Could be done by creating a new StaticMatrix type.

What if we take another vector `q` and consider 
```
  ∂(q·p) / ∂c
```
Then this is quite unambiguously a `Vector{SVector}`, 
```
   ∂(q·p) / ∂ci_b = sum_a q_a Bi * (a==b) = q_b B_i. 
```
