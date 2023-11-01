module SamplingHelper

export update!, makeSampler!

using Random
using ..SpinSystems
using ..SingleSpinFlip
using ..MultiSpinFlip
using ..OnBipartiteGraph

"""
    update!(ua::SingleSpinUpdatingAlgorithm; [rng=default_rng()])

Update a spin of `ua.spinSystem.spinConfiguration`.
`rng` is used to get a random number when the updating algorithm depends on randomness.
This method forwards `ua` to another `update!(ua, updatedNode, fluctuation)` method with substituting default values into `updatedNode` and `fluctuation`.

# Arguments
- `ua::SingleSpinUpdatingAlgorithm`: A spin system with parameters.
- `rng::AbstractRNG`: A random number generator.
"""
function update!(ua::SingleSpinFlip.SingleSpinUpdatingAlgorithm; rng::AbstractRNG=Random.default_rng())
    updatedNode = rand(rng, eachindex(getSpinConfiguration(ua)))
    fluctuation = rand(rng, ua.distribution)
    SingleSpinFlip.update!(ua, updatedNode, fluctuation)
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
    update!(ua::MultiSpinUpdatingAlgorithm; [rng=default_rng()])

Update a spin of `ua.spinSystem.spinConfiguration`.
`rng` is used to get a random number when the updating algorithm depends on randomness.
This method forwards `ua` to another `update!(ua, updatedNode, fluctuation)` method with substituting default values into `updatedNode` and `fluctuation`.

# Arguments
- `ua::MultiSpinUpdatingAlgorithm`: A spin system with parameters.
- `rng::AbstractRNG`: A random number generator.
"""
function update!(ua::MultiSpinFlip.MultiSpinUpdatingAlgorithm; rng::AbstractRNG=Random.default_rng())
    fluctuation = rand(rng, ua.distribution)
    MultiSpinFlip.update!(ua, fluctuation)
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
    update!(ua::UpdatingAlgorithmOnBipartiteGraph; [rng=default_rng()])

Update a spin of `ua.spinSystem.spinConfiguration`.
`rng` is used to get a random number when the updating algorithm depends on randomness.
This method forwards `ua` to another `update!(ua, updatedNode, fluctuation)` method with substituting default values into `updatedNode` and `fluctuation`.

# Arguments
- `ua::UpdatingAlgorithmOnBipartiteGraph`: A spin system on a bipartite graph with parameters.
- `rng::AbstractRNG`: A random number generator.
"""
function update!(ua::OnBipartiteGraph.UpdatingAlgorithmOnBipartiteGraph; rng::AbstractRNG=Random.default_rng())
    fluctuationForSpinConfiguration = rand(rng, ua.distribution, length(ua.spinSystem.spinConfiguration))
    fluctuationForHiddenLayer = rand(rng, ua.distribution, length(ua.spinSystem.hiddenLayer))
    OnBipartiteGraph.update!(ua, fluctuationForSpinConfiguration, fluctuationForHiddenLayer)
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
