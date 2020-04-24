#!/bin/bash
cd ~/.julia/dev
if [ -d "SHIPs_gh" ]
then
    echo "SHIPs_gh already exists."
else
    echo "SHIPs_gh does not yet exists - cloneing from git repo."
    git clone https://github.com/JuliaMolSim/SHIPs.jl.git SHIPs_gh
fi
cd SHIPs_gh
echo "make sure gh-pages is checked out"
git checkout gh-pages
echo "obtain doc build from ~/.julia/SHIPs/doc/build"
cp -R ../SHIPs/docs/build/* ./dev/
echo "commit changes"
git add dev/*
git commit -a -m "update online docs"
echo "push to gh-pages"
git push origin gh-pages
cd ~/.julia/dev/SHIPs/docs
