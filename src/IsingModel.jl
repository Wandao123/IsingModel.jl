module IsingModel

export SpinSystem
export getSpinConfiguration, getCouplingCoefficients, getExternalMagneticField
export calcEnergy, update!
export AsynchronousHopfieldNetwork
export GlauberDynamics
export MetropolisMethod

using LinearAlgebra
using Random, Distributions

@enum IsingSpin DownSpin = -1 UpSpin = +1

# HACK: How to receive SparseMatrixCSC as a couping-coefficients matrix?
#       "MethodError: no method matching zero(::Type{Any})" is caused if passing without converting one by the `collect` method.
mutable struct SpinSystem
    spinConfiguration::AbstractVector{Number}
    couplingCoefficients::AbstractMatrix{Number}  # Like an adjacency matrix
    externalMagneticField::AbstractVector{Number}  # Bias on each site

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

getSpinConfiguration(s::UpdatingAlgorithm)::AbstractVector{Number} = s.spinSystem.spinConfiguration
getCouplingCoefficients(s::UpdatingAlgorithm)::AbstractMatrix{AbstractFloat} = s.spinSystem.couplingCoefficients
getExternalMagneticField(s::UpdatingAlgorithm)::AbstractVector{AbstractFloat} = s.spinSystem.externalMagneticField

function calcEnergy(spinSystem::SpinSystem)::AbstractFloat
    return -0.5 * spinSystem.spinConfiguration' * spinSystem.couplingCoefficients * spinSystem.spinConfiguration
        - spinSystem.externalMagneticField' * spinSystem.spinConfiguration
end

calcEnergy(s::UpdatingAlgorithm)::AbstractFloat = calcEnergy(s.spinSystem)

function calcLocalMagneticField(spinSystem::SpinSystem)::AbstractVector{AbstractFloat}
    return spinSystem.couplingCoefficients * spinSystem.spinConfiguration
        + spinSystem.externalMagneticField
end

function calcLocalMagneticField(spinSystem::SpinSystem, nodeIndex::Int)::AbstractFloat
    return spinSystem.couplingCoefficients[nodeIndex, :]' * spinSystem.spinConfiguration
        + spinSystem.externalMagneticField[nodeIndex]
end

calcLocalMagneticField(s::UpdatingAlgorithm)::AbstractVector{AbstractFloat} = calcLocalMagneticField(s.spinSystem)
calcLocalMagneticField(s::UpdatingAlgorithm, x::Int)::AbstractFloat = calcLocalMagneticField(s.spinSystem, x)

mutable struct AsynchronousHopfieldNetwork <: UpdatingAlgorithm
    spinSystem::SpinSystem
end

function update!(s::AsynchronousHopfieldNetwork)
    x = rand(eachindex(getSpinConfiguration(s)))  # An updated node label
    getSpinConfiguration(s)[x] = ifelse(
            getCouplingCoefficients(s)[x, :]' * getSpinConfiguration(s)
                - getExternalMagneticField(s)[x]
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

    x = rand(eachindex(getSpinConfiguration(s)))  # An updated node label
    getSpinConfiguration(s)[x] = sign(
            2 * calcLocalMagneticField(s, x)
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

    x = rand(eachindex(getSpinConfiguration(s)))  # An updated node label
    getSpinConfiguration(s)[x] = sign(
            2 * calcLocalMagneticField(s, x)
            - rand(Exponential()) * s.temperature * getSpinConfiguration(s)[x]
        ) |> Int
end

end
