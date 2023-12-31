### A Pluto.jl notebook ###
# v0.19.30

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
	using IsingModel.SpinSystems: calcEnergy
	using IsingModel.SamplingHelper: makeSampler!
	using BenchmarkTools
	using Distributions
	import Graphs
	using GraphPlot: gplot
	using LinearAlgebra
	using Plots: histogram, plot, plot!
	using Random
	using Revise
end

# ╔═╡ 4f999a32-2d07-45c7-a30f-2f87c65a053a
md"""
# Demonstrations
"""

# ╔═╡ 9fb8f983-0fae-4fbd-bf6f-3a788323f25f
const N = 9  # The number of nodes

# ╔═╡ 3b95e78f-0743-4a3e-b334-2ced85113b89
# Generate a square lattice with the periodic boundary condition.
begin
	const SIDE_LENGTH = (Int ∘ ceil ∘ sqrt)(N)

	# A direct way making the adjacency matrix
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
	G = Graphs.grid((SIDE_LENGTH, SIDE_LENGTH), periodic=true)
	adjacencyMatrix = map(Graphs.adjacency_matrix(G)) do c
		ifelse(c == 0, 0, -1)
	end
end

# ╔═╡ d956b7b4-1bc2-41c7-95e2-8e8baef41c7f
gplot(G)

# ╔═╡ 2d3c39e0-c31b-4741-bc85-4d00082f64c4
bias = zeros(N)

# ╔═╡ 1fdecb38-6c88-4c15-bd62-836e77b0170a
const INITIAL_CONFIGURATION = 2 .* rand(Bernoulli(0.5), N) .- 1

# ╔═╡ 9539349c-1cdb-4bc0-8d9e-94406a96e49c
begin
	spinSystem = SpinSystems.SpinSystem(
		INITIAL_CONFIGURATION,
		adjacencyMatrix,
		bias
	)
	println("Hamiltonian = $(calcEnergy(spinSystem))")
	pinningParameter = 0.5 * eigmax(collect(adjacencyMatrix))
	spinSystemOnBipartiteGraph = SpinSystems.SpinSystemOnBipartiteGraph(
		INITIAL_CONFIGURATION,
		INITIAL_CONFIGURATION,
		0.5 * (adjacencyMatrix + pinningParameter * I),
		0.5 * bias,
		0.5 * bias
	)
	println("Modified Hamiltonian = $(calcEnergy(spinSystemOnBipartiteGraph) + 0.5 * pinningParameter * N)")
end

# ╔═╡ 5caf6d8b-8e0a-4e49-9ada-d7a403ca04de
begin
	const MAX_STEPS = N^2 ÷ 2
	const INITIAL_TEMPERATURE = float(MAX_STEPS)
	const FINAL_TEMPERATURE = 0.0

	#annealingSchedule(n) = (FINAL_TEMPERATURE - INITIAL_TEMPERATURE) / MAX_STEPS * n + INITIAL_TEMPERATURE
	annealingSchedule(n) = INITIAL_TEMPERATURE ^ (-n / MAX_STEPS)

	function runAnnealer(
		Algorithm::Type{<:SpinSystems.UpdatingAlgorithm},
		spinSystem::SpinSystems.SpinSystem,
		INITIAL_TEMPERATURE::Float64=-1.0
	)::Vector{Float64}
		if INITIAL_TEMPERATURE < 0.0
			return map(
				calcEnergy,
				makeSampler!(
					Algorithm(deepcopy(spinSystem)),
					MAX_STEPS,
					annealingSchedule=annealingSchedule
				)
			)
		else
			return map(
				calcEnergy,
				makeSampler!(
					Algorithm(deepcopy(spinSystem), INITIAL_TEMPERATURE),
					MAX_STEPS,
					annealingSchedule=annealingSchedule
				)
			)
		end
	end

	function runAnnealer(
		Algorithm::Type{<:SpinSystems.UpdatingAlgorithmOnBipartiteGraph},
		spinSystem::SpinSystems.SpinSystemOnBipartiteGraph,
		INITIAL_TEMPERATURE::Float64=-1.0
	)::Vector{Float64}
		return map(
			a -> calcEnergy(a) + 0.5 * pinningParameter * N,
			makeSampler!(
				Algorithm(deepcopy(spinSystem), INITIAL_TEMPERATURE),
				MAX_STEPS,
				annealingSchedule=annealingSchedule
			)
		)
	end
end

# ╔═╡ b14cb2a9-d69f-48eb-a268-6bca0ff91675
@benchmark runAnnealer(SingleSpinFlip.GlauberDynamics, spinSystem, INITIAL_TEMPERATURE)

# ╔═╡ 34e062e0-e878-4f48-a7fd-91009b83be0f
begin
	plot(xlabel="MC steps", ylabel="Energy")
	plot!(runAnnealer(SingleSpinFlip.AsynchronousHopfieldNetwork, spinSystem), label="Hopfield")
	plot!(runAnnealer(SingleSpinFlip.GlauberDynamics, spinSystem, INITIAL_TEMPERATURE), label="Glauber")
	plot!(runAnnealer(SingleSpinFlip.MetropolisMethod, spinSystem, INITIAL_TEMPERATURE), label="Metropolis")
	plot!(runAnnealer(OnBipartiteGraph.StochasticCellularAutomata, spinSystemOnBipartiteGraph, INITIAL_TEMPERATURE), label="SCA")
	plot!(runAnnealer(OnBipartiteGraph.MomentumAnnealing, spinSystemOnBipartiteGraph, INITIAL_TEMPERATURE), label="MA")
end

# ╔═╡ eb0ec036-1315-4c36-b0e2-55e60081e2c5
begin
	histogram(
		map(
			(binary -> parse(Int, binary, base=2))
				∘ join
				∘ (spin -> (1 .- spin) .÷ 2) 
				∘ SpinSystems.getSpinConfiguration,
			makeSampler!(SingleSpinFlip.GlauberDynamics(spinSystem, N / 4), 50000)
		),
		normalize=:pdf
	)
	plot!(xlabel="Spin configuration", ylabel="Frequency")
end

# ╔═╡ Cell order:
# ╟─4f999a32-2d07-45c7-a30f-2f87c65a053a
# ╠═3db0f7f0-26b8-11ee-345a-970b2c1cf2ed
# ╠═9fb8f983-0fae-4fbd-bf6f-3a788323f25f
# ╠═3b95e78f-0743-4a3e-b334-2ced85113b89
# ╠═d956b7b4-1bc2-41c7-95e2-8e8baef41c7f
# ╠═2d3c39e0-c31b-4741-bc85-4d00082f64c4
# ╠═1fdecb38-6c88-4c15-bd62-836e77b0170a
# ╠═9539349c-1cdb-4bc0-8d9e-94406a96e49c
# ╠═5caf6d8b-8e0a-4e49-9ada-d7a403ca04de
# ╠═b14cb2a9-d69f-48eb-a268-6bca0ff91675
# ╠═34e062e0-e878-4f48-a7fd-91009b83be0f
# ╠═eb0ec036-1315-4c36-b0e2-55e60081e2c5
