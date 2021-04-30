
# ACE.jl Documentation

This package implements approximation schemes for permutation and isometry invariant functions, with focus on modelling atomic interactions. It provides constructions of symmetric polynomial bases, imposing permutation and isometry invariance. Heavy use is made of trigonometric polynomials and spherical harmonics to obtain rotation invariance.


```@contents
Pages = ["intro.md",
         "gettingstarted.md",
         "devel.md",
         "polyproducts.md"]
Depth = 1
```

### Editing and Building the Documentation Locally

* To build the documentation locally, use the `make.jl` script. Simply switch to `ACE/docs` and execute `julia make.jl`
* To publish the documentation to github, use the `publish.sh` script: switch to `ACE/docs` and execute `./publish.sh`. This will close the `ACE_gh repository (if it isn't already), the copy into it the website, and git commit, push.
