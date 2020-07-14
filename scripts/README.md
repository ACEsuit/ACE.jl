

# SHIPs/scripts

This directory contains a range of scripts for a variety of uses.


## Regression Tests

### Manual regression tests

1. Checkout a commit containing a suitable `regressions.jl` script
2. Copy `regressions.jl` to `myreg.jl` or a suitable name
3. Edit the script to change the settings, make sure the new script is not checked in
4. Generate the regression tests by running `julia myreg.jl`. If the current commit should be added to the regression tests then instead run `julia myreg.jl label` where `label` is a label for the current commit.
5. checkout the next commit that should be tested and run `julia myreg.jl label2`, iterate.

For example
```
git checkout v0.7.x
julia -O3 myreg.jl head    # generate tests and test the latest commit
git checkout v0.7.0        # test the latest commit
julia -O3 myreg.jl v0.7.0
git checkout v0.6.5        # test an older commit
julia -O3 myreg.jl v0.6.5
git checkout v0.7.x        # to go back to where we started
```

The regression script will store the version of Julia and the machine that is used. In particular it is possible to use different versions not only of the package but also of Julia.

In the future it will also be possible to use this machinery to test multi-threaded performance.



### Auto regression tests

Most of the time the regression tests will always be the same. In this case the shell script `autoregressions.jl` can be used.

```
autoregressions j15 v0.7.0 v0.6.5
```
will run exactly the sequence of commands above, with a few safety checks.s



### Analysing the regression data
