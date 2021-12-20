
### States Implementation

An abtstract particle/atom/... is described by a state variable. Normally, 
it would be of type `State`, which is simply a wrapper around a `NamedTuple`, 
and largely behaves as such except that various operations are overwritten
without committed type piracy. 

There is also a `DState` which is formally the derivative of a `State`. For 
example if `x = State(rr = ..., x = ...)` and `p` some property, then the 
derivative of `p` w.r.t. `x` will be a `DState`. We can infer the `DState{...}` type corresponding to a `State{...}` type using `dstate_type`.

#### Ways to construct a `State`

The two standard constructors are 
* `State(nt::NamedTuple)` : this is the obvious one.
* `State(; kwargs) = State(NamedTuple(kwargs...))` : wrapper for slightly nicer notation

Then there are a few variations on those that are only needed 
internally. 


#### Printing States 

the function `show` is implemented which gives a somewhat visually sensible 
printout of a state. Use `ACE.showdigits!` to change how many digits are 
printed. 

