
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

#### Construct a DState or DState type 

* The constructors available for `State` also work for `DState`
* In order to construct the type of a `DState` corresponding to a `State`, one can use `dstate_type`; this will create a `DState` type including exactly those symbols present in `State` corresponding to continuos property variables.

#### Printing States 

the function `show` is implemented which gives a somewhat visually sensible 
printout of a state. Use `ACE.showdigits!` to change how many digits are 
printed. 

#### Arithmetic

Quite a lot of arithmetic is defined on States and DStates to make the behave more or less like vectors. This is a bit of an abuse, technically speaking. Only DStates are vectors, while States are *points*. Oh well... 

E.g., we have 
* `+, -, *`
* `real, imag, complex`
* `rand, randn, zero`
* `dot, norm, contract, sumsq`
* `isapprox`
If further functionality is needed, it can be easily added. 

#### More internals 

Both `State` and `DState` simply wrap a `NamedTuple`. To access them we can just use `X.rr, X.u` etc, which uses an overloaded `getproperty`. To get access to the named tuple itself, use 
* `_x(X)`
To get access to the tupe parameters of the named tuple, use 
* `_syms(X)`
* `_tt(X)`
* `_symstt(X)`
The (indices of) the continuous parameters can be extracted using 
* `_findcts`
* `_ctssyms`


