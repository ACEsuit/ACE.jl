


using Documenter, ACE

makedocs(sitename="ACE.jl Documentation",
         pages = [
        "Home" => "index.md",
        "Getting Started" => "gettingstarted.md",
        "Math" => "math.md",
        "Developer Docs" => "devel.md",
        "Types & Functions" => "docs.md"
         ])

        #  https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104
                 
deploydocs(
    repo = "github.com/ACEsuit/ACE.jl.git",
    devbranch = "main"
)
