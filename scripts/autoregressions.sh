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
echo "Run regressions.jl at HEAD which is at $HEAD"

for var in "$@"
do
   echo "Checkout $var"
   git checkout $var
   echo "Run regressions.jl at $var"
   $JULIA regressions.jl
done

git checkout $HEAD
