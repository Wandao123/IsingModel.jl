module IsingModel

export SpinSystem
export calcEnergy
export AsynchronousHopfieldNetwork
export GlauberDynamics
export MetropolisMethod
export update!

using LinearAlgebra
using Random, Distributions

@enum IsingSpin DownSpin = -1 UpSpin = +1

mutable struct SpinSystem
    spinConfiguration::Vector{Number}
    couplingCoefficients::Matrix{AbstractFloat}  # Like an adjacency matrix
    externalMagneticField::Vector{AbstractFloat}  # Bias on each site

    function SpinSystem(spinConfiguration, couplingCoefficients, externalMagneticField)
        numNodes = length(spinConfiguration)
        (row, column) = size(couplingCoefficients)
        if row != column
            error("The coupling-coefficient matrix is not a square matrix: $(row)rows ≠ $(column)columns.")
            return nothing
        elseif numNodes < row
            @warn "The size of the spin-configuration vector is too smaller than the size of the coupling-coefficient matrix.  The incorresponding components of the coupling-coefficient matrix are ignored."
            couplingCoefficients = couplingCoefficients[1:numNodes, 1:numNodes]
        elseif numNodes > row
            @warn "The size of the spin-configuration vector is too bigger than the size of the coupling-coefficient matrix.  The incorresponding components of the spin-configuration vector are ignored."
            spinConfiguration = spinConfiguration[1:row]
        elseif !issymmetric(couplingCoefficients)
            @warn "The coupling-coefficient matrix should be symmetric.  It is symmetrized by its upper-triangular components automatically."
            couplingCoefficients = Symmetric(couplingCoefficients, :U)
        end
        if any(diag(couplingCoefficients) .!= 0)
            @warn "The diagonal components of the coupling-coefficient matrix should be zero.  Their non-zero components are ignored."
            couplingCoefficients -= Diagonal(couplingCoefficients)
        end
        numBias = length(externalMagneticField)
        if row != numBias
            error("The size of the coupling-coefficient matrix does not match the size of the external-magnetic-field vector: $(row) ≠ $(numBias).")
            return nothing
        elseif numNodes < numBias
            @warn "The size of the spin-configuration vector is too smaller than the size of the external-magnetic-field vector.  The incorresponding components of the external-magnetic-field vector are ignored."
            externalMagneticField = externalMagneticField[1:numNodes]
        elseif numNodes > numBias
            @warn "The size of the spin-configuration vector is too bigger than the size of the external-magnetic-field vector.  The incorresponding components of the spin-configuration vector are ignored."
            spinConfiguration = spinConfiguration[1:numBias]
        end
        return new(spinConfiguration, couplingCoefficients, externalMagneticField)
    end
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
    s.spinSystem.spinConfiguration[x] = ifelse(
            # HACK: Should we point out the index x before calculating matrices?
            (s.spinSystem.couplingCoefficients * s.spinSystem.spinConfiguration
            - s.spinSystem.externalMagneticField)[x]
            >= 0,
            +1,
            -1
        )
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
