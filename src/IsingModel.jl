module IsingModel

export SpinSystem
export calcEnergy
export AsynchronousHopfieldNetwork
export GlauberDynamics
export MetropolisMethod
export update!

using Random, Distributions

@enum IsingSpin DownSpin = -1 UpSpin = +1

mutable struct SpinSystem
    spinConfiguration::Vector{Number}
    couplingCoefficients::Matrix{AbstractFloat}  # Like an adjacency matrix
    externalMagneticField::Vector{AbstractFloat}  # Bias on each site
end

"""
    UpdatingAlgorithm

Suppose that any sub-struct of this type has the spinSystem::SpinSystem field.
"""
abstract type UpdatingAlgorithm end

function calcEnergy(spinSystem::SpinSystem)::AbstractFloat
    return -0.5 * spinSystem.spinConfiguration' * spinSystem.couplingCoefficients * spinSystem.spinConfiguration
        - spinSystem.externalMagneticField' * spinSystem.spinConfiguration
end

calcEnergy(s::UpdatingAlgorithm)::AbstractFloat = calcEnergy(s.spinSystem)

function calcLocalMagneticField(spinSystem::SpinSystem)::Vector{AbstractFloat}
    return spinSystem.couplingCoefficients * spinSystem.spinConfiguration
        + spinSystem.externalMagneticField
end

calcLocalMagneticField(s::UpdatingAlgorithm)::Vector{AbstractFloat} = calcLocalMagneticField(s.spinSystem)

mutable struct AsynchronousHopfieldNetwork <: UpdatingAlgorithm
    spinSystem::SpinSystem
end

function update!(s::AsynchronousHopfieldNetwork)
    x = rand(eachindex(s.spinSystem.spinConfiguration))  # An updated node label
    s.spinSystem.spinConfiguration[x] = sign(
            # HACK: Should we point out the index x before calculating matrices?
            (s.spinSystem.couplingCoefficients * s.spinSystem.spinConfiguration
            - s.spinSystem.externalMagneticField)[x]
        ) |> Int
end

mutable struct GlauberDynamics <: UpdatingAlgorithm
    spinSystem::SpinSystem
    temperature::AbstractFloat
end

function update!(s::GlauberDynamics)
    if s.temperature < 0
        @warn "$temperature is negative."
    end

    x = rand(eachindex(s.spinSystem.spinConfiguration))  # An updated node label
    s.spinSystem.spinConfiguration[x] = sign(
            2 * calcLocalMagneticField(s)[x]
            - rand(Logistic()) * s.temperature
        ) |> Int
end

mutable struct MetropolisMethod <: UpdatingAlgorithm
    spinSystem::SpinSystem
    temperature::AbstractFloat
end

function update!(s::MetropolisMethod)
    if s.temperature < 0
        @warn "$temperature is negative."
    end

    x = rand(eachindex(s.spinSystem.spinConfiguration))  # An updated node label
    s.spinSystem.spinConfiguration[x] = sign(
            2 * calcLocalMagneticField(s)[x]
            - rand(Exponential()) * s.temperature * s.spinSystem.spinConfiguration[x]
        ) |> Int
end

end
