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

echo "Run regressions at HEAD which is at $HEAD; this also generates the tests"
$JULIA -O3 --color=yes _4u10r39s_.jl

for var in "$@"
do
   echo "Checkout $var"
   git checkout -q $var
   echo "Run regressions.jl at $var"
   $JULIA -O3 --color=yes _4u10r39s_.jl
done

git checkout -q $HEAD
rm _4u10r39s_.jl
