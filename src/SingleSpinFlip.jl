module SingleSpinFlip

export update!
export AsynchronousHopfieldNetwork, GlauberDynamics, MetropolisMethod

using LinearAlgebra
using Random, Distributions
using ..SpinSystems

abstract type SingleSpinUpdatingAlgorithm <: UpdatingAlgorithm end

mutable struct AsynchronousHopfieldNetwork <: SingleSpinUpdatingAlgorithm
    spinSystem::SpinSystem
    distribution::ContinuousUnivariateDistribution  # Dummy

    AsynchronousHopfieldNetwork(spinSystem::SpinSystem) = new(spinSystem, Uniform())
end

"""
    update!(s, updatedNode, fluctuation)

Update a spin of `s.spinSystem.spinConfiguration` at a site `updatedNode`.
`fluctuation` is used by random updating algorithms.
When the algorithm is deterministic, it is regarded as a dummy parameter to unify the interface of each algorithm.

# Arguments
- `s::T <: SingleSpinUpdatingAlgorithm`: A spin system with parameters.
- `updatedNode::Integer`: An updated node label..
- `fluctuation::AbstractFloat`: A random number which obeys `s.distribution` (required).
"""
function update!(s::AsynchronousHopfieldNetwork, updatedNode::Integer, ::AbstractFloat=0.0)
    s.spinSystem.spinConfiguration[updatedNode] = ifelse(
        getCouplingCoefficients(s)[updatedNode, :]' * getSpinConfiguration(s) - getExternalMagneticField(s)[updatedNode] >= 0,
        +1,
        -1
    )
end

mutable struct GlauberDynamics <: SingleSpinUpdatingAlgorithm
    spinSystem::SpinSystem
    temperature::AbstractFloat
    distribution::ContinuousUnivariateDistribution

    GlauberDynamics(spinSystem::SpinSystem, temperature::AbstractFloat) = new(spinSystem, temperature, Logistic())
end

function update!(s::GlauberDynamics, updatedNode::Integer, fluctuation::AbstractFloat)
    if s.temperature < 0
        @warn "$temperature is negative."
    end

    s.spinSystem.spinConfiguration[updatedNode] = sign(
        2 * calcLocalMagneticField(s, updatedNode)
        -
        fluctuation * s.temperature
    ) |> Int
end

mutable struct MetropolisMethod <: SingleSpinUpdatingAlgorithm
    spinSystem::SpinSystem
    temperature::AbstractFloat
    distribution::ContinuousUnivariateDistribution

    MetropolisMethod(spinSystem::SpinSystem, temperature::AbstractFloat) = new(spinSystem, temperature, Exponential())
end

function update!(s::MetropolisMethod, updatedNode::Integer, fluctuation::AbstractFloat)
    if s.temperature < 0
        @warn "$temperature is negative."
    end

    s.spinSystem.spinConfiguration[updatedNode] = sign(
        2 * calcLocalMagneticField(s, updatedNode)
        -
        fluctuation * s.temperature * getSpinConfiguration(s)[updatedNode]
    ) |> Int
end

end
