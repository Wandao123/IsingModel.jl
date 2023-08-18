module OnBipartiteGraph

export update!, makeSampler!
export StochasticCellularAutomata

using LinearAlgebra
using Random, Distributions
using ..SpinSystems

"""
    update!(s::UpdatingAlgorithmOnBipartiteGraph; [rng=default_rng()])

Update a spin of `s.spinSystem.spinConfiguration`.
`rng` is used to get a random number when the updating algorithm depends on randomness.
This method forwards `s` to another `update!(s, updatedNode, fluctuation)` method with substituting default values into `updatedNode` and `fluctuation`.

# Arguments
- `s::UpdatingAlgorithmOnBipartiteGraph`: A spin system on a bipartite graph with parameters.
- `rng::AbstractRNG`: A random number generator.
"""
function update!(s::UpdatingAlgorithmOnBipartiteGraph; rng::AbstractRNG=Random.default_rng())
    fluctuationForSpinConfiguration = rand(rng, s.distribution, length(s.spinSystem.spinConfiguration))
    fluctuationForHiddenLayer = rand(rng, s.distribution, length(s.spinSystem.hiddenLayer))
    update!(s, fluctuationForSpinConfiguration, fluctuationForHiddenLayer)
end

function makeSampler!(updatingAlgorithm::UpdatingAlgorithmOnBipartiteGraph, maxMCSteps::Integer; annealingSchedule::Function=n -> updatingAlgorithm.temperature, rng::AbstractRNG=Random.default_rng())::Channel{UpdatingAlgorithmOnBipartiteGraph}
    if maxMCSteps < 0
        @warn "$maxMCSteps is negative."
    end
    function decreaseTemperature(updatingAlgorithm, stepCounter)
        if hasproperty(updatingAlgorithm, :temperature)
            updatingAlgorithm.temperature = annealingSchedule(stepCounter)
        else
            nothing
        end
    end
    fluctuationsForSpinConfiguration = rand(rng, updatingAlgorithm.distribution, (length(getSpinConfiguration(updatingAlgorithm)), maxMCSteps))
    fluctuationsForHiddenLayer = rand(rng, updatingAlgorithm.distribution, (length(getHiddenLayer(updatingAlgorithm)), maxMCSteps))

    Channel{UpdatingAlgorithmOnBipartiteGraph}() do channel
        decreaseTemperature(updatingAlgorithm, 0)
        put!(channel, updatingAlgorithm)
        for stepCounter = 1:maxMCSteps
            decreaseTemperature(updatingAlgorithm, stepCounter)
            update!(updatingAlgorithm, fluctuationsForSpinConfiguration[:, stepCounter], fluctuationsForHiddenLayer[:, stepCounter])
            put!(channel, updatingAlgorithm)
        end
    end
end

mutable struct StochasticCellularAutomata <: UpdatingAlgorithmOnBipartiteGraph
    spinSystem::SpinSystemOnBipartiteGraph
    temperature::AbstractFloat
    distribution::ContinuousUnivariateDistribution

    StochasticCellularAutomata(spinSystem::SpinSystemOnBipartiteGraph, temperature::AbstractFloat) = new(spinSystem, temperature, Logistic())
end

"""
    update!(s, fluctuationForSpinConfiguration, fluctuationForHiddenLayer)

Update a spin of `s.spinSystem.spinConfiguration`.
`fluctuationForSpinConfiguration` and `fluctuationForHiddenLayer` are used by random updating algorithms.
When the algorithm is deterministic, they are regarded as a dummy parameter to unify the interface of each algorithm.

# Arguments
- `s::T <: UpdatingAlgorithmOnBipartiteGraph`: A spin system on a bipartite graph with parameters.
- `fluctuationForSpinConfiguration::AbstractVector{AbstractFloat}`: A random vector whose each component obeys `s.distribution` (required).  Its size must be the same as `s.spinConfiguration`.
- `fluctuationForHiddenLayer::AbstractVector{AbstractFloat}`: A random vector whose each component obeys `s.distribution` (required).  Its size must be the same as `s.hiddenLayer`.
"""
function update!(s::StochasticCellularAutomata, fluctuationForSpinConfiguration::AbstractVector{<:AbstractFloat}, fluctuationForHiddenLayer::AbstractVector{<:AbstractFloat})
    if s.temperature < 0
        @warn "$temperature is negative."
    end

    s.spinSystem.hiddenLayer = (Int ∘ sign).(
            2 * calcLocalAuxiliaryBias(s)
            - fluctuationForHiddenLayer * s.temperature
        )
    s.spinSystem.spinConfiguration = (Int ∘ sign).(
            2 * calcLocalMagneticField(s)
            - fluctuationForSpinConfiguration * s.temperature
        )
end

end
