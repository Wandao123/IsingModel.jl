### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 3db0f7f0-26b8-11ee-345a-970b2c1cf2ed
# Before running this demonstration, install python libraries via REPL:
# `julia> using Conda`
# `julia> Conda.add("scipy")`
# `julia> Conda.add("networkx")`
# Ref: https://stackoverflow.com/questions/70974016/julia-pluto-cannot-find-dev-installed-package
begin
	import Pkg
	# activate the shared project environment
	Pkg.activate(Base.current_project())
	# instantiate, i.e. make sure that all packages are downloaded
	Pkg.instantiate()

	using IsingModel
	using BenchmarkTools
	using Conda
	using Graphs, GraphPlot
	using LinearAlgebra
	using PyCall
	using Random, Distributions
	using Revise

	# Uncomment the following two lines the first time to run this demonstration.
	#Conda.add("scipy")
	#Conda.add("networkx")
	nx = pyimport("networkx")
end

# ╔═╡ 9fb8f983-0fae-4fbd-bf6f-3a788323f25f
const N = 64  # The number of nodes

# ╔═╡ 3b95e78f-0743-4a3e-b334-2ced85113b89
# Generate a square lattice with the periodic boundary condition.
begin
	sideLength = (Int ∘ ceil ∘ sqrt)(N)

	# Directly making the adjacency matrix
	#=adjacencyMatrix = zeros((N, N))
	for i = 0:N-1
		if i % sideLength < sideLength - 1 && i + 1 < N
			adjacencyMatrix[i + 1, (i + 1) + 1] = 1
		else
			adjacencyMatrix[(i ÷ sideLength) * sideLength + 1, i + 1] = 1
		end
		if i ÷ sideLength < (N - 1) ÷ sideLength
			adjacencyMatrix[i + 1, i + sideLength + 1] = 1
		else
			adjacencyMatrix[i % sideLength + 1, i + 1] = 1
		end
	end
	adjacencyMatrix = Symmetric(adjacencyMatrix, :U)=#

	display(collect(adjacency_matrix(grid((sideLength, sideLength)))))  # Another way

	G = nx.grid_2d_graph(sideLength, sideLength, periodic=true)
	nx.set_edge_attributes(G, values=-1, name="weight")
	adjacencyMatrix = nx.adjacency_matrix(G).todense()
end

# ╔═╡ 2d3c39e0-c31b-4741-bc85-4d00082f64c4
bias = zeros(N)

# ╔═╡ 1fdecb38-6c88-4c15-bd62-836e77b0170a
const initialConfiguration = 2 .* rand(Bernoulli(0.5), N) .- 1

# ╔═╡ 9539349c-1cdb-4bc0-8d9e-94406a96e49c
spinSystem = SpinSystem(initialConfiguration, adjacencyMatrix, bias)

# ╔═╡ 5caf6d8b-8e0a-4e49-9ada-d7a403ca04de
begin
	const initialTemperature = 10.0
	const finalTemperature = 0.0
	const maxSteps = N^2

	annealingSchedule(n) = (finalTemperature - initialTemperature) / maxSteps * n + initialTemperature
end

# ╔═╡ cc79b855-1f14-49b0-b115-4fbc7bde505e
@benchmark begin
	#algorithm = AsynchronousHopfieldNetwork(deepcopy(spinSystem))
	algorithm = GlauberDynamics(deepcopy(spinSystem), initialTemperature)
	#algorithm = MetropolisMethod(deepcopy(spinSystem), initialTemperature)
	data = zeros(maxSteps + 1)
	for n in 0:maxSteps
		algorithm.temperature = annealingSchedule(n)
		update!(algorithm)
		data[n + 1] = calcEnergy(algorithm)
	end
	println(data)
end

# ╔═╡ b14cb2a9-d69f-48eb-a268-6bca0ff91675
@benchmark begin
	algorithm = GlauberDynamics(deepcopy(spinSystem), initialTemperature)
	#algorithm = MetropolisMethod(deepcopy(spinSystem), initialTemperature)
	println(map(calcEnergy, takeSamples!(algorithm, maxSteps, annealingSchedule)))
end

# ╔═╡ Cell order:
# ╠═3db0f7f0-26b8-11ee-345a-970b2c1cf2ed
# ╠═9fb8f983-0fae-4fbd-bf6f-3a788323f25f
# ╠═3b95e78f-0743-4a3e-b334-2ced85113b89
# ╠═2d3c39e0-c31b-4741-bc85-4d00082f64c4
# ╠═1fdecb38-6c88-4c15-bd62-836e77b0170a
# ╠═9539349c-1cdb-4bc0-8d9e-94406a96e49c
# ╠═5caf6d8b-8e0a-4e49-9ada-d7a403ca04de
# ╠═cc79b855-1f14-49b0-b115-4fbc7bde505e
# ╠═b14cb2a9-d69f-48eb-a268-6bca0ff91675
