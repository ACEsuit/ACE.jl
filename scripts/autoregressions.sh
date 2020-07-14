#!/bin/bash

# check that the git repository is clean
if ! [ -z "$(git status --untracked-files=no --porcelain)" ]; then
  echo ERROR: abort since working directory has changed tracked files.
  # exit 1
fi

# save the name of the current branch
HEAD=$(git rev-parse --abbrev-ref HEAD)
JULIA=$1
shift

echo "Using Julia binary $JULIA"
echo "Make temporary copy of regressions.jl"
cp regressions.jl _4u10r39s_.jl

echo "Generate the tests"
$JULIA -O3 --color=yes --project=.. generate_regression_tests.jl

echo "Run regression tests at HEAD which is at $HEAD"
$JULIA -O3 --color=yes --project=.. _4u10r39s_.jl head

for var in "$@"
do
   echo "Checkout $var"
   git checkout -q $var
   echo "Run regression tests at $var"
   $JULIA -O3 --color=yes --project=.. _4u10r39s_.jl $var
done

git checkout -q $HEAD
rm _4u10r39s_.jl
