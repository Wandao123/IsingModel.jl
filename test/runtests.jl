using IsingModel
using Random
using SparseArrays
using Test

function runAnnealer(updatingAlgorithm, makeSampler!)
    rng = Random.default_rng()
    Random.seed!(rng, 128)
    sampler = makeSampler!(updatingAlgorithm, 5, annealingSchedule=n -> 10.0^(-n), rng=rng)
    result = nothing
    while isopen(sampler)
        result = take!(sampler)
        println(result)  # "Channel is closed" exception occurs if commenting out this line.
    end
    return result.spinSystem.spinConfiguration
end

@testset "SingleSpinFlip" begin
    s = SpinSystem([-1, +1], sparse([0 1; 1 0]), [0, 0])
    @test calcEnergy(s) == 1.0
    @test runAnnealer(AsynchronousHopfieldNetwork(deepcopy(s)), makeSampler!) == [1, 1]
    @test runAnnealer(GlauberDynamics(deepcopy(s), 10.0), makeSampler!) == [1, 1]
    @test runAnnealer(MetropolisMethod(deepcopy(s), 10.0), makeSampler!) == [1, 1]
end

@testset "OnBipartiteGraph" begin
    s = OnBipartiteGraph.SpinSystemOnBipartiteGraph([-1, +1], [-1, +1, -1], sparse([1 1 1; 1 1 1]), [0, 0], [0, 0, 0])
    @test OnBipartiteGraph.calcEnergy(s) == 0.0
    @test runAnnealer(OnBipartiteGraph.StochasticCellularAutomata(deepcopy(s), 10.0), OnBipartiteGraph.makeSampler!) == [1, 1]
end
