
# an example script on how to run regression tests 
# best do it manually for now, until we learn how to use 
# this well and then setup an automated workflow 

using ACE, PkgBenchmark 

results = benchmarkpkg(ACE)
writeresults(@__DIR__() * "/results.json", results)

results_main = benchmarkpkg(ACE, "main")
writeresults(@__DIR__() * "/results_main.json", results)

results = readresults(@__DIR__() * "/results.json")
results_base = readresults(@__DIR__() * "/results_main.json")
judgement = judge(results, results_main)
export_markdown(@__DIR__() * "/judge.md", judgement)
