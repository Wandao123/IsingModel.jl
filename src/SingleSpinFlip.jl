module SingleSpinFlip

export update!
export AsynchronousHopfieldNetwork, GlauberDynamics, MetropolisMethod

using LinearAlgebra
using Random, Distributions
using ..SpinSystems

H1(ua::UpdatingAlgorithm, weight::AbstractFloat) = convert(eltype(getSpinConfiguration(ua)), SpinSystems.heaviside(weight))

abstract type SingleSpinUpdatingAlgorithm <: UpdatingAlgorithm end

mutable struct AsynchronousHopfieldNetwork <: SingleSpinUpdatingAlgorithm
    spinSystem::SpinSystem
    distribution::ContinuousUnivariateDistribution  # Dummy

    AsynchronousHopfieldNetwork(spinSystem::SpinSystem) = new(spinSystem, Uniform())
end

"""
    update!(ua, updatedNode, fluctuation)

Update a spin of `ua.spinSystem.spinConfiguration` at a site `updatedNode`.
`fluctuation` is used by random updating algorithms.
When the algorithm is deterministic, it is regarded as a dummy parameter to unify the interface of each algorithm.

# Arguments
- `ua::T <: SingleSpinUpdatingAlgorithm`: A spin system with parameters.
- `updatedNode::Integer`: An updated node label..
- `fluctuation::AbstractFloat`: A random number which obeys `ua.distribution` (required).
"""
function update!(ua::AsynchronousHopfieldNetwork, updatedNode::Integer, ::AbstractFloat=0.0)
    ua.spinSystem.spinConfiguration[updatedNode] = 2 * H1(
        ua,
        getCouplingCoefficients(ua)[updatedNode, :]' * getSpinConfiguration(ua)
        - getExternalMagneticField(ua)[updatedNode]
    ) - 1
end

mutable struct GlauberDynamics <: SingleSpinUpdatingAlgorithm
    spinSystem::SpinSystem
    temperature::AbstractFloat
    distribution::ContinuousUnivariateDistribution

    GlauberDynamics(spinSystem::SpinSystem, temperature::AbstractFloat) = new(spinSystem, temperature, Logistic())
end

function update!(ua::GlauberDynamics, updatedNode::Integer, fluctuation::AbstractFloat)
    if ua.temperature < 0
        @warn "$temperature is negative."
    end

    ua.spinSystem.spinConfiguration[updatedNode] = 2 * H1(
        ua,
        2 * calcLocalMagneticField(ua, updatedNode)
        - fluctuation * ua.temperature
    ) - 1
end

mutable struct MetropolisMethod <: SingleSpinUpdatingAlgorithm
    spinSystem::SpinSystem
    temperature::AbstractFloat
    distribution::ContinuousUnivariateDistribution

    MetropolisMethod(spinSystem::SpinSystem, temperature::AbstractFloat) = new(spinSystem, temperature, Exponential())
end

function update!(ua::MetropolisMethod, updatedNode::Integer, fluctuation::AbstractFloat)
    if ua.temperature < 0
        @warn "$temperature is negative."
    end

    ua.spinSystem.spinConfiguration[updatedNode] = 2 * H1(
        ua,
        2 * calcLocalMagneticField(ua, updatedNode)
        - fluctuation * ua.temperature * getSpinConfiguration(ua)[updatedNode]
    ) - 1
end

end
