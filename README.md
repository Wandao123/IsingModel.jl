# IsingModel

[![Build Status](https://github.com/Wandao123/IsingModel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Wandao123/IsingModel.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Wandao123.github.io/IsingModel.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Wandao123.github.io/IsingModel.jl/dev)
[![Coverage](https://codecov.io/gh/Wandao123/IsingModel.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Wandao123/IsingModel.jl)

## Getting started

### Demonstration

You can see a demonstration in [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Wandao123/IsingModel.jl/HEAD?labpath=demo.ipynb).
It may take some time.

### Generating adjacency matrices by Python's NetworkX library

Before running this demonstration, please install python libraries via REPL:
```
julia> using Conda
julia> Conda.add("scipy")
julia> Conda.add("networkx")
```
Depending on your OS, you may be required to chenge the path to Python interpreter and build PyCall.
```
julia> ENV["PYTHON"] = ENV["HOME"] * "/.julia/conda/3/x86_64/bin/python3"
julia> using Pkg
julia> Pkg.build("PyCall")
```
See also [the README file of PyCall](https://github.com/JuliaPy/PyCall.jl).
