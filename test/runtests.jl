using IsingModel
using Random
using SparseArrays
using Test

function runAnnealer(updatingAlgorithm)
    rng = Random.default_rng()
    Random.seed!(rng, 128)
    sampler = SamplingHelper.makeSampler!(updatingAlgorithm, 5, annealingSchedule=n -> 10.0^(-n), rng=rng)
    result = nothing
    while isopen(sampler)
        result = take!(sampler)
        println(result)  # "Channel is closed" exception occurs if commenting out this line.
    end
    SamplingHelper.update!(updatingAlgorithm)
    return result.spinSystem.spinConfiguration
end

@testset "SingleSpinFlip" begin
    ss = SpinSystems.SpinSystem([-1, +1], sparse([0 1; 1 0]), [0, 0])
    @test SpinSystems.calcEnergy(ss) == 1.0
    @test runAnnealer(SingleSpinFlip.AsynchronousHopfieldNetwork(deepcopy(ss))) ∈ [[1, 1], [-1, -1]]
    @test runAnnealer(SingleSpinFlip.GlauberDynamics(deepcopy(ss), 10.0)) ∈ [[1, 1], [-1, -1]]
    @test runAnnealer(SingleSpinFlip.MetropolisMethod(deepcopy(ss), 10.0)) ∈ [[1, 1], [-1, -1]]
end

@testset "OnBipartiteGraph" begin
    ss = OnBipartiteGraph.SpinSystemOnBipartiteGraph([-1, +1], [-1, +1, -1], sparse([1 1 1; 1 1 1]), [0, 0], [0, 0, 0])
    @test OnBipartiteGraph.calcEnergy(ss) == 0.0
    @test runAnnealer(OnBipartiteGraph.StochasticCellularAutomata(deepcopy(ss), 10.0)) ∈ [[1, 1], [-1, -1]]
    @test runAnnealer(OnBipartiteGraph.MomentumAnnealing(deepcopy(ss), 10.0)) ∈ [[1, 1], [-1, -1]]
end
