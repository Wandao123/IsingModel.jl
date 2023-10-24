### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 3db0f7f0-26b8-11ee-345a-970b2c1cf2ed
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
	#using Graphs, GraphPlot
	using LinearAlgebra
	using Plots
	using PyCall
	using Random, Distributions
	using Revise
	using SparseArrays

	# Uncomment the following two lines the first time to run this demonstration.
	#Conda.add("scipy")
	#Conda.add("networkx")
	const nx = pyimport("networkx")
end

# ╔═╡ 4f999a32-2d07-45c7-a30f-2f87c65a053a
md"""
# Demonstrations
"""

# ╔═╡ 8c380ce4-4e41-43d9-afb2-e2252f58e5e4
# Ref: https://github.com/JuliaPy/PyCall.jl/issues/204
begin
	const scipy_sparse_find = pyimport("scipy.sparse")["find"]
	function mysparse(Apy::PyObject)
		IA, JA, SA = scipy_sparse_find(Apy)
		return sparse(Int[i+1 for i in IA], Int[i+1 for i in JA], SA)
	end
end

# ╔═╡ 9fb8f983-0fae-4fbd-bf6f-3a788323f25f
const N = 64  # The number of nodes

# ╔═╡ 3b95e78f-0743-4a3e-b334-2ced85113b89
# Generate a square lattice with the periodic boundary condition.
begin
	const SIDE_LENGTH = (Int ∘ ceil ∘ sqrt)(N)

	# Directly making the adjacency matrix
	#=adjacencyMatrix = zeros((N, N))
	for i = 0:N-1
		if i % SIDE_LENGTH < SIDE_LENGTH - 1 && i + 1 < N
			adjacencyMatrix[i + 1, (i + 1) + 1] = 1
		else
			adjacencyMatrix[(i ÷ SIDE_LENGTH) * SIDE_LENGTH + 1, i + 1] = 1
		end
		if i ÷ SIDE_LENGTH < (N - 1) ÷ SIDE_LENGTH
			adjacencyMatrix[i + 1, i + SIDE_LENGTH + 1] = 1
		else
			adjacencyMatrix[i % SIDE_LENGTH + 1, i + 1] = 1
		end
	end
	adjacencyMatrix = Symmetric(adjacencyMatrix, :U)=#

	# A way using the Graphs.jl library
	#adjacencyMatrix = adjacency_matrix(grid((SIDE_LENGTH, SIDE_LENGTH), periodic=true))

	G = nx.grid_2d_graph(SIDE_LENGTH, SIDE_LENGTH, periodic=true)
	nx.set_edge_attributes(G, values=-1, name="weight")
	adjacencyMatrix = mysparse(nx.adjacency_matrix(G))
end

# ╔═╡ 2d3c39e0-c31b-4741-bc85-4d00082f64c4
bias = zeros(N)

# ╔═╡ 1fdecb38-6c88-4c15-bd62-836e77b0170a
const INITIAL_CONFIGURATION = 2 .* rand(Bernoulli(0.5), N) .- 1

# ╔═╡ 9539349c-1cdb-4bc0-8d9e-94406a96e49c
spinSystem = SpinSystem(INITIAL_CONFIGURATION, adjacencyMatrix, bias)

# ╔═╡ 5caf6d8b-8e0a-4e49-9ada-d7a403ca04de
begin
	const MAX_STEPS = N ^ 2
	const INITIAL_TEMPERATURE = float(MAX_STEPS)
	const FINAL_TEMPERATURE = 0.0

	#annealingSchedule(n) = (FINAL_TEMPERATURE - INITIAL_TEMPERATURE) / MAX_STEPS * n + INITIAL_TEMPERATURE
	annealingSchedule(n) = INITIAL_TEMPERATURE ^ (-n)

	function runAnnealer(Algorithm::Type{<:IsingModel.UpdatingAlgorithm}, spinSystem::SpinSystem, INITIAL_TEMPERATURE=-1.0)::Vector{Float64}
	    if INITIAL_TEMPERATURE < 0.0
	        return map(calcEnergy, takeSamples!(Algorithm(deepcopy(spinSystem)), MAX_STEPS, annealingSchedule))
	    else
	        return map(calcEnergy, takeSamples!(Algorithm(deepcopy(spinSystem), INITIAL_TEMPERATURE), MAX_STEPS, annealingSchedule))
    	end
	end
end

# ╔═╡ b14cb2a9-d69f-48eb-a268-6bca0ff91675
@benchmark runAnnealer(GlauberDynamics, spinSystem, INITIAL_TEMPERATURE)

# ╔═╡ 34e062e0-e878-4f48-a7fd-91009b83be0f
begin
	plot(xlabel="MC steps", ylabel="Energy")
	plot!(runAnnealer(AsynchronousHopfieldNetwork, spinSystem), label="Hopfield")
	plot!(runAnnealer(GlauberDynamics, spinSystem, INITIAL_TEMPERATURE), label="Glauber")
	plot!(runAnnealer(MetropolisMethod, spinSystem, INITIAL_TEMPERATURE), label="Metropolis")
end

# ╔═╡ Cell order:
# ╟─4f999a32-2d07-45c7-a30f-2f87c65a053a
# ╠═3db0f7f0-26b8-11ee-345a-970b2c1cf2ed
# ╠═8c380ce4-4e41-43d9-afb2-e2252f58e5e4
# ╠═9fb8f983-0fae-4fbd-bf6f-3a788323f25f
# ╠═3b95e78f-0743-4a3e-b334-2ced85113b89
# ╠═2d3c39e0-c31b-4741-bc85-4d00082f64c4
# ╠═1fdecb38-6c88-4c15-bd62-836e77b0170a
# ╠═9539349c-1cdb-4bc0-8d9e-94406a96e49c
# ╠═5caf6d8b-8e0a-4e49-9ada-d7a403ca04de
# ╠═b14cb2a9-d69f-48eb-a268-6bca0ff91675
# ╠═34e062e0-e878-4f48-a7fd-91009b83be0f
