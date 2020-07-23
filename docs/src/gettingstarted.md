
# Getting Started

(this is an early draft of this document; please send me comments!)

## Installation

1. Install [Julia](https://julialang.org), the latest versions of `ACE.jl` require `v1.3` but should also work ok with `v1.4`, future versions will likely require `v1.4`.
2. Install the [`MolSim` registry](https://github.com/JuliaMolSim/MolSim); from the Julia REPL, switch to package manager `]` and then run
```julia
registry add https://github.com/JuliaMolSim/MolSim.git
```
3. Install some important registered packages; from Julia REPL / package manager:
```julia
add PyCall IJulia     # add more important packages from General registry
add JuLIP ACE ASE   # maybe add other packages from MolSim registry
```
4. For fitting, need to install also [`IPFitting.jl`](https://github.com/cortner/IPFitting.jl),
```julia
add IPFitting
```
(Keep fingers crossed and hope it will be compatible with the current version of `ACE.jl`...)


## Workflow
