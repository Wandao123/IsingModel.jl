module IsingModel

export SpinSystem
export getSpinConfiguration, getCouplingCoefficients, getExternalMagneticField
export calcEnergy, update!, makeSampler!
export AsynchronousHopfieldNetwork, GlauberDynamics, MetropolisMethod

using LinearAlgebra
using Random, Distributions

@enum IsingSpin DownSpin = -1 UpSpin = +1

mutable struct SpinSystem
    spinConfiguration::AbstractVector{<:Number}
    couplingCoefficients::AbstractMatrix{<:AbstractFloat}  # An weighted adjacency matrix
    externalMagneticField::AbstractVector{<:AbstractFloat}  # Bias on each site

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
        return new(spinConfiguration, float.(couplingCoefficients), float.(externalMagneticField))
    end
end

"""
    UpdatingAlgorithm

Suppose that any sub-struct of this type has the spinSystem::SpinSystem field.
"""
abstract type UpdatingAlgorithm end

getSpinConfiguration(s::UpdatingAlgorithm)::AbstractVector{<:Number} = s.spinSystem.spinConfiguration
getCouplingCoefficients(s::UpdatingAlgorithm)::AbstractMatrix{<:AbstractFloat} = s.spinSystem.couplingCoefficients
getExternalMagneticField(s::UpdatingAlgorithm)::AbstractVector{<:AbstractFloat} = s.spinSystem.externalMagneticField

function calcEnergy(spinSystem::SpinSystem)::AbstractFloat
    return -0.5 * spinSystem.spinConfiguration' * spinSystem.couplingCoefficients * spinSystem.spinConfiguration
        - spinSystem.externalMagneticField' * spinSystem.spinConfiguration
end

calcEnergy(s::UpdatingAlgorithm)::AbstractFloat = calcEnergy(s.spinSystem)

function calcLocalMagneticField(spinSystem::SpinSystem)::AbstractVector{AbstractFloat}
    return spinSystem.couplingCoefficients * spinSystem.spinConfiguration
        + spinSystem.externalMagneticField
end

function calcLocalMagneticField(spinSystem::SpinSystem, nodeIndex::Integer)::AbstractFloat
    return spinSystem.couplingCoefficients[nodeIndex, :]' * spinSystem.spinConfiguration
        + spinSystem.externalMagneticField[nodeIndex]
end

calcLocalMagneticField(s::UpdatingAlgorithm)::AbstractVector{AbstractFloat} = calcLocalMagneticField(s.spinSystem)
calcLocalMagneticField(s::UpdatingAlgorithm, x::Integer)::AbstractFloat = calcLocalMagneticField(s.spinSystem, x)

abstract type SingleSpinUpdatingAlgorithm <: UpdatingAlgorithm end

"""
    update!(s::SingleSpinUpdatingAlgorithm; [rng=default_rng()])

Update a spin of `s.spinSystem.spinConfiguration`.
`rng` is used to get a random number when the updating algorithm depends on randomness.
This method forwards `s` to another `update!(s, updatedNode, fluctuation)` method with substituting default values into `updatedNode` and `fluctuation`.

# Arguments
- `s::SingleSpinUpdatingAlgorithm`: A spin system with parameters.
- `rng::AbstractRNG`: A random number generator.
"""
function update!(s::SingleSpinUpdatingAlgorithm; rng::AbstractRNG=Random.default_rng())
    updatedNode = rand(rng, eachindex(getSpinConfiguration(s)))
    fluctuation = rand(rng, s.distribution)
    update!(s, updatedNode, fluctuation)
end

function makeSampler!(updatingAlgorithm::SingleSpinUpdatingAlgorithm, maxMCSteps::Integer; annealingSchedule::Function=n -> updatingAlgorithm.temperature, rng::AbstractRNG=Random.default_rng())::Channel{UpdatingAlgorithm}
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

    Channel{UpdatingAlgorithm}() do channel
        put!(channel, updatingAlgorithm)
        for stepCounter = 1:maxMCSteps
            decreaseTemperature(updatingAlgorithm, stepCounter)
            update!(updatingAlgorithm, updatedNodes[stepCounter], fluctuations[stepCounter])
            put!(channel, updatingAlgorithm)
        end
    end
end

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
            getCouplingCoefficients(s)[updatedNode, :]' * getSpinConfiguration(s)
                - getExternalMagneticField(s)[updatedNode]
                >= 0,
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
            - fluctuation * s.temperature
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
            - fluctuation * s.temperature * getSpinConfiguration(s)[updatedNode]
        ) |> Int
end

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
        put!(channel, updatingAlgorithm)
        for stepCounter = 1:maxMCSteps
            decreaseTemperature(updatingAlgorithm, stepCounter)
            update!(updatingAlgorithm, fluctuations[stepCounter])
            put!(channel, updatingAlgorithm)
        end
    end
end

export OnBipartiteGraph
include("IsingModelOnBipartiteGraph.jl")

end
