
# Preliminary AD Capabilities

Our aim is to make ACE.jl fully differentiable. Some first steps in this direction are now complete providing some initial AD capabilities. This page provides a preliminary documentation and records some limitations and pitfalls. 

Example:  
```julia 
# initialize the linear ACE with two invariant properties 
c_m = rand(SVector{2,Float64}, length(basis))
model = ACE.LinearACEModel(basis, c_m)
# wrap a nonlinearity around it 
FS = p -> sum( (1 .+ val.(p).^2).^0.5 )
fsmodel = cfg -> FS(evaluate(model, cfg))
# AD it to get the forces
grad_fsmodel = cfg -> Zygote.gradient(fsmodel, cfg)[1]  

# now define some loss that uses model values and model gradients 
y = randn(SVector{3, Float64}, length(cfg))
loss = model -> sum( sum(abs2, g.rr - y) 
                     for (g, y) in zip(grad_fsmodel(model, cfg), y) )
# and we can differentiate this w.r.t the parameters
Zygote.gradient(loss, model)[1]
```

Remarks: 
* `val` : `evaluate(model, cfg)` will return an `SVector` containing two `Invariant`s. To extract an actual value from that, we use `val` which is simply defined as `val(x) = x.val`. The point though is that we also defined adjoints for `val` which propagage through the differentiation. This is why `FS` uses `val` in its definition. 
* Especially the AD capabilities of ACE.jl are very much a draft without much concern for performance. 
* composition of ACE models is not supported yet, but hopefully coming. 
* Nobody knows what will happen in the above example if the linear ace model produces covariant instead of invariant properties :). This is all untested and will likely break. Please file issues. 
* There are a few places that are still "hacks", see TODOs in the main ACE.jl code. 
