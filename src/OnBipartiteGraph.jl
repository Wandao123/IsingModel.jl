module OnBipartiteGraph

export update!, makeSampler!
export StochasticCellularAutomata

using LinearAlgebra
using Random, Distributions
using ..SpinSystems

mutable struct StochasticCellularAutomata <: UpdatingAlgorithmOnBipartiteGraph
    spinSystem::SpinSystemOnBipartiteGraph
    temperature::AbstractFloat
    distribution::ContinuousUnivariateDistribution

    StochasticCellularAutomata(spinSystem::SpinSystemOnBipartiteGraph, temperature::AbstractFloat) = new(spinSystem, temperature, Logistic())
end

"""
    update!(ua, fluctuationForSpinConfiguration, fluctuationForHiddenLayer)

Update a spin of `ua.spinSystem.spinConfiguration`.
`fluctuationForSpinConfiguration` and `fluctuationForHiddenLayer` are used by random updating algorithms.
When the algorithm is deterministic, they are regarded as a dummy parameter to unify the interface of each algorithm.

# Arguments
- `ua::T <: UpdatingAlgorithmOnBipartiteGraph`: A spin system on a bipartite graph with parameters.
- `fluctuationForSpinConfiguration::AbstractVector{AbstractFloat}`: A random vector whose each component obeys `ua.distribution` (required).  Its size must be the same as `ua.spinConfiguration`.
- `fluctuationForHiddenLayer::AbstractVector{AbstractFloat}`: A random vector whose each component obeys `ua.distribution` (required).  Its size must be the same as `ua.hiddenLayer`.
"""
function update!(ua::StochasticCellularAutomata, fluctuationForSpinConfiguration::AbstractVector{<:AbstractFloat}, fluctuationForHiddenLayer::AbstractVector{<:AbstractFloat})
    if ua.temperature < 0
        @warn "$temperature is negative."
    end

    ua.spinSystem.hiddenLayer = 2 * SpinSystems.heaviside.(
        2 * calcLocalAuxiliaryBias(ua)
        - fluctuationForHiddenLayer * ua.temperature
    ) .- 1
    ua.spinSystem.spinConfiguration = 2 * SpinSystems.heaviside.(
        2 * calcLocalMagneticField(ua)
        - fluctuationForSpinConfiguration * ua.temperature
    ) .- 1
end

mutable struct MomentumAnnealing <: UpdatingAlgorithmOnBipartiteGraph
    spinSystem::SpinSystemOnBipartiteGraph
    temperature::AbstractFloat
    distribution::ContinuousUnivariateDistribution

    MomentumAnnealing(spinSystem::SpinSystemOnBipartiteGraph, temperature::AbstractFloat) = new(spinSystem, temperature, Exponential())
end

function update!(ua::MomentumAnnealing, fluctuationForSpinConfiguration::AbstractVector{<:AbstractFloat}, fluctuationForHiddenLayer::AbstractVector{<:AbstractFloat})
    if ua.temperature < 0
        @warn "$temperature is negative."
    end

    ua.spinSystem.hiddenLayer = 2 * SpinSystems.heaviside.(
        2 * calcLocalAuxiliaryBias(ua)
        - fluctuationForHiddenLayer * ua.temperature .* getHiddenLayer(ua)
    ) .- 1
    ua.spinSystem.spinConfiguration = 2 * SpinSystems.heaviside.(
        2 * calcLocalMagneticField(ua)
        - fluctuationForSpinConfiguration * ua.temperature .* getSpinConfiguration(ua)
    ) .- 1
end

end
