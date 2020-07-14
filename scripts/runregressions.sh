#!/bin/bash

JULIA="/Users/ortner/gits/julia15/julia -O3 --color=yes --project=.."

# check that the git repository is clean
if ! [ -z "$(git status --untracked-files=no --porcelain)" ]; then
  echo ERROR: abort since working directory has changed tracked files.
  exit 1
fi

# save the name of the current branch
HEAD=$(git rev-parse --abbrev-ref HEAD)

echo "Using Julia binary $JULIA"
echo "Make temporary copy of regressions.jl"
cp regressions.jl _4u10r39s_.jl
REGSCRIPT="_4u10r39s_.jl"

echo "Generate the tests"
$JULIA generate_regression_tests.jl

echo "Run regression tests at HEAD which is at $HEAD"
$JULIA $REGSCRIPT head

var="v0.7.0"
echo "Run regression tests at $var"
git checkout -q $var
$JULIA -e 'using Pkg; Pkg.pin(name="JuLIP", version="0.9.7"); Pkg.resolve()'
$JULIA _4u10r39s_.jl $var

echo "Return to original head"
git checkout -q $HEAD
rm _4u10r39s_.jl
