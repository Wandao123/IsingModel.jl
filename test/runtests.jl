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
    s = SpinSystems.SpinSystem([-1, +1], sparse([0 1; 1 0]), [0, 0])
    @test SpinSystems.calcEnergy(s) == 1.0
    @test runAnnealer(SingleSpinFlip.AsynchronousHopfieldNetwork(deepcopy(s))) == [1, 1]
    @test runAnnealer(SingleSpinFlip.GlauberDynamics(deepcopy(s), 10.0)) == [1, 1]
    @test runAnnealer(SingleSpinFlip.MetropolisMethod(deepcopy(s), 10.0)) == [1, 1]
end

@testset "OnBipartiteGraph" begin
    s = OnBipartiteGraph.SpinSystemOnBipartiteGraph([-1, +1], [-1, +1, -1], sparse([1 1 1; 1 1 1]), [0, 0], [0, 0, 0])
    @test OnBipartiteGraph.calcEnergy(s) == 0.0
    @test runAnnealer(OnBipartiteGraph.StochasticCellularAutomata(deepcopy(s), 10.0)) == [1, 1]
    @test runAnnealer(OnBipartiteGraph.MomentumAnnealing(deepcopy(s), 10.0)) == [1, 1]
end
