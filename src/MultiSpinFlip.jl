module MultiSpinFlip

export update!, makeSampler!

using LinearAlgebra
using Random, Distributions
using ..SpinSystems

abstract type MultiSpinUpdatingAlgorithm <: UpdatingAlgorithm end

"""
    update!(s::MultiSpinUpdatingAlgorithm; [rng=default_rng()])

Update a spin of `s.spinSystem.spinConfiguration`.
`rng` is used to get a random number when the updating algorithm depends on randomness.
This method forwards `s` to another `update!(s, updatedNode, fluctuation)` method with substituting default values into `updatedNode` and `fluctuation`.

# Arguments
- `s::MultiSpinUpdatingAlgorithm`: A spin system with parameters.
- `rng::AbstractRNG`: A random number generator.
"""
function update!(s::MultiSpinUpdatingAlgorithm; rng::AbstractRNG=Random.default_rng())
    fluctuation = rand(rng, s.distribution)
    update!(s, fluctuation)
end

function makeSampler!(updatingAlgorithm::MultiSpinUpdatingAlgorithm, maxMCSteps::Integer, annealingSchedule::Function=n -> updatingAlgorithm.temperature; rng::AbstractRNG=Random.default_rng())::Channel{UpdatingAlgorithm}
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

    Channel{UpdatingAlgorithm}() do channel
        decreaseTemperature(updatingAlgorithm, 0)
        put!(channel, updatingAlgorithm)
        for stepCounter = 1:maxMCSteps
            decreaseTemperature(updatingAlgorithm, stepCounter)
            update!(updatingAlgorithm, fluctuations[stepCounter])
            put!(channel, updatingAlgorithm)
        end
    end
end

end
