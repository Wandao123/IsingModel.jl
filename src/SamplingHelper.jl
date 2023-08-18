module SamplingHelper

export update!, makeSampler!

using Random
using ..SpinSystems
using ..SingleSpinFlip
using ..MultiSpinFlip
using ..OnBipartiteGraph

"""
    update!(s::SingleSpinUpdatingAlgorithm; [rng=default_rng()])

Update a spin of `s.spinSystem.spinConfiguration`.
`rng` is used to get a random number when the updating algorithm depends on randomness.
This method forwards `s` to another `update!(s, updatedNode, fluctuation)` method with substituting default values into `updatedNode` and `fluctuation`.

# Arguments
- `s::SingleSpinUpdatingAlgorithm`: A spin system with parameters.
- `rng::AbstractRNG`: A random number generator.
"""
function update!(s::SingleSpinFlip.SingleSpinUpdatingAlgorithm; rng::AbstractRNG=Random.default_rng())
    updatedNode = rand(rng, eachindex(getSpinConfiguration(s)))
    fluctuation = rand(rng, s.distribution)
    SingleSpinFlip.update!(s, updatedNode, fluctuation)
end

function makeSampler!(updatingAlgorithm::SingleSpinFlip.SingleSpinUpdatingAlgorithm, maxMCSteps::Integer; annealingSchedule::Function=n -> updatingAlgorithm.temperature, rng::AbstractRNG=Random.default_rng())::Channel{SpinSystems.UpdatingAlgorithm}
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
    updatedNodes = rand(rng, eachindex(getSpinConfiguration(updatingAlgorithm)), maxMCSteps)
    fluctuations = rand(rng, updatingAlgorithm.distribution, maxMCSteps)

    Channel{SpinSystems.UpdatingAlgorithm}() do channel
        decreaseTemperature(updatingAlgorithm, 0)
        put!(channel, updatingAlgorithm)
        for stepCounter = 1:maxMCSteps
            decreaseTemperature(updatingAlgorithm, stepCounter)
            SingleSpinFlip.update!(updatingAlgorithm, updatedNodes[stepCounter], fluctuations[stepCounter])
            put!(channel, updatingAlgorithm)
        end
    end
end

"""
    update!(s::MultiSpinUpdatingAlgorithm; [rng=default_rng()])

Update a spin of `s.spinSystem.spinConfiguration`.
`rng` is used to get a random number when the updating algorithm depends on randomness.
This method forwards `s` to another `update!(s, updatedNode, fluctuation)` method with substituting default values into `updatedNode` and `fluctuation`.

# Arguments
- `s::MultiSpinUpdatingAlgorithm`: A spin system with parameters.
- `rng::AbstractRNG`: A random number generator.
"""
function update!(s::MultiSpinFlip.MultiSpinUpdatingAlgorithm; rng::AbstractRNG=Random.default_rng())
    fluctuation = rand(rng, s.distribution)
    MultiSpinFlip.update!(s, fluctuation)
end

function makeSampler!(updatingAlgorithm::MultiSpinFlip.MultiSpinUpdatingAlgorithm, maxMCSteps::Integer, annealingSchedule::Function=n -> updatingAlgorithm.temperature; rng::AbstractRNG=Random.default_rng())::Channel{SpinSystems.UpdatingAlgorithm}
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
    fluctuations = rand(rng, updatingAlgorithm.distribution, maxMCSteps)

    Channel{SpinSystems.UpdatingAlgorithm}() do channel
        decreaseTemperature(updatingAlgorithm, 0)
        put!(channel, updatingAlgorithm)
        for stepCounter = 1:maxMCSteps
            decreaseTemperature(updatingAlgorithm, stepCounter)
            MultiSpinFlip.update!(updatingAlgorithm, fluctuations[stepCounter])
            put!(channel, updatingAlgorithm)
        end
    end
end

"""
    update!(s::UpdatingAlgorithmOnBipartiteGraph; [rng=default_rng()])

Update a spin of `s.spinSystem.spinConfiguration`.
`rng` is used to get a random number when the updating algorithm depends on randomness.
This method forwards `s` to another `update!(s, updatedNode, fluctuation)` method with substituting default values into `updatedNode` and `fluctuation`.

# Arguments
- `s::UpdatingAlgorithmOnBipartiteGraph`: A spin system on a bipartite graph with parameters.
- `rng::AbstractRNG`: A random number generator.
"""
function update!(s::OnBipartiteGraph.UpdatingAlgorithmOnBipartiteGraph; rng::AbstractRNG=Random.default_rng())
    fluctuationForSpinConfiguration = rand(rng, s.distribution, length(s.spinSystem.spinConfiguration))
    fluctuationForHiddenLayer = rand(rng, s.distribution, length(s.spinSystem.hiddenLayer))
    OnBipartiteGraph.update!(s, fluctuationForSpinConfiguration, fluctuationForHiddenLayer)
end

function makeSampler!(updatingAlgorithm::OnBipartiteGraph.UpdatingAlgorithmOnBipartiteGraph, maxMCSteps::Integer; annealingSchedule::Function=n -> updatingAlgorithm.temperature, rng::AbstractRNG=Random.default_rng())::Channel{SpinSystems.UpdatingAlgorithmOnBipartiteGraph}
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

    Channel{SpinSystems.UpdatingAlgorithmOnBipartiteGraph}() do channel
        decreaseTemperature(updatingAlgorithm, 0)
        put!(channel, updatingAlgorithm)
        for stepCounter = 1:maxMCSteps
            decreaseTemperature(updatingAlgorithm, stepCounter)
            OnBipartiteGraph.update!(updatingAlgorithm, fluctuationsForSpinConfiguration[:, stepCounter], fluctuationsForHiddenLayer[:, stepCounter])
            put!(channel, updatingAlgorithm)
        end
    end
end

end
