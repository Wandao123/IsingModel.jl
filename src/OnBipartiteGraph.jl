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

mutable struct MomentumAnnealing <: UpdatingAlgorithmOnBipartiteGraph
    spinSystem::SpinSystemOnBipartiteGraph
    temperature::AbstractFloat
    distribution::ContinuousUnivariateDistribution

    MomentumAnnealing(spinSystem::SpinSystemOnBipartiteGraph, temperature::AbstractFloat) = new(spinSystem, temperature, Exponential())
end

function update!(s::MomentumAnnealing, fluctuationForSpinConfiguration::AbstractVector{<:AbstractFloat}, fluctuationForHiddenLayer::AbstractVector{<:AbstractFloat})
    if s.temperature < 0
        @warn "$temperature is negative."
    end

    s.spinSystem.hiddenLayer = (Int ∘ sign).(
            2 * calcLocalAuxiliaryBias(s)
            - fluctuationForHiddenLayer * s.temperature .* getHiddenLayer(s)
        )
    s.spinSystem.spinConfiguration = (Int ∘ sign).(
            2 * calcLocalMagneticField(s)
            - fluctuationForSpinConfiguration * s.temperature .* getSpinConfiguration(s)
        )
end

end
