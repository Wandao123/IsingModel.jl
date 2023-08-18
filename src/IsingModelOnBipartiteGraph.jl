module OnBipartiteGraph

export SpinSystemOnBipartiteGraph
export getSpinConfiguration, getHiddenLayer, getCouplingCoefficients, getExternalMagneticField, getAuxiliaryBias
export calcEnergy, update!, makeSampler!
export StochasticCellularAutomata

using LinearAlgebra
using Random, Distributions

mutable struct SpinSystemOnBipartiteGraph
    spinConfiguration::AbstractVector{<:Number}
    hiddenLayer::AbstractVector{<:Number}
    couplingCoefficients::AbstractMatrix{<:AbstractFloat}  # An weighted adjacency matrix with shape [# of visible nodes] × [# of hidden nodes]
    externalMagneticField::AbstractVector{<:AbstractFloat}
    auxiliaryBias::AbstractVector{<:AbstractFloat}

    function SpinSystemOnBipartiteGraph(spinConfiguration, hiddenLayer, couplingCoefficients, externalMagneticField, auxiliaryBias)
        numVisibleNodes = length(spinConfiguration)
        numHiddenNodes = length(hiddenLayer)
        (row, column) = size(couplingCoefficients)
        if row != numVisibleNodes
            error("The size of the coupling-coefficient matrix does not match the number of visible and hidden nodes: $(numVisibleNodes)nodes ≠ $(row)rows.")
            return nothing
        elseif column != numHiddenNodes
            error("The size of the coupling-coefficient matrix does not match the number of visible and hidden nodes: $(numHiddenNodes)nodes ≠ $(column)columns.")
            return nothing
        end
        numFields = length(externalMagneticField)
        numBias = length(auxiliaryBias)
        if numFields != numVisibleNodes
            error("The size of the external-magnetic-field vector does not match the number of visible nodes: $(numFields) ≠ $(numVisibleNodes).")
            return nothing
        elseif numBias != numHiddenNodes
            error("The size of the eauxiliary-bias vector does not match the number of hidden nodes: $(numBias) ≠ $(numHiddenNodes).")
            return nothing
        end
        return new(spinConfiguration, hiddenLayer, float.(couplingCoefficients), float.(externalMagneticField), float.(auxiliaryBias))
    end
end

"""
    UpdatingAlgorithmOnBipartiteGraph

Suppose that any sub-struct of this type has the spinSystem::SpinSystemOnBipartiteGraph field.
"""
abstract type UpdatingAlgorithmOnBipartiteGraph end

getSpinConfiguration(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractVector{<:Number} = s.spinSystem.spinConfiguration
getHiddenLayer(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractVector{<:Number} = s.spinSystem.hiddenLayer
getCouplingCoefficients(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractMatrix{<:AbstractFloat} = s.spinSystem.couplingCoefficients
getExternalMagneticField(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractVector{<:AbstractFloat} = s.spinSystem.externalMagneticField
getAuxiliaryBias(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractVector{<:AbstractFloat} = s.spinSystem.auxiliaryBias

function calcEnergy(spinSystem::SpinSystemOnBipartiteGraph)::AbstractFloat
    return -spinSystem.spinConfiguration' * spinSystem.couplingCoefficients * spinSystem.hiddenLayer
        - spinSystem.externalMagneticField' * spinSystem.spinConfiguration
        - spinSystem.auxiliaryBias' * spinSystem.hiddenLayer
end

calcEnergy(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractFloat = calcEnergy(s.spinSystem)

function calcLocalMagneticField(spinSystem::SpinSystemOnBipartiteGraph)::AbstractVector{AbstractFloat}
    return spinSystem.couplingCoefficients * spinSystem.hiddenLayer
        + spinSystem.externalMagneticField
end

calcLocalMagneticField(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractVector{AbstractFloat} = calcLocalMagneticField(s.spinSystem)

function calcLocalAuxiliaryBias(spinSystem::SpinSystemOnBipartiteGraph)::AbstractVector{AbstractFloat}
    return spinSystem.couplingCoefficients' * spinSystem.spinConfiguration
        + spinSystem.auxiliaryBias
end

calcLocalAuxiliaryBias(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractVector{AbstractFloat} = calcLocalAuxiliaryBias(s.spinSystem)

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
