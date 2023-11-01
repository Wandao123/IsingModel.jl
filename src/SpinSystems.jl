module SpinSystems

export SpinSystem, UpdatingAlgorithm
export getSpinConfiguration, getCouplingCoefficients, getExternalMagneticField
export calcEnergy, calcLocalMagneticField
export SpinSystemOnBipartiteGraph, UpdatingAlgorithmOnBipartiteGraph
export getHiddenLayer, getAuxiliaryBias
export calcLocalAuxiliaryBias

using LinearAlgebra

################ For general graphs ################

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
setSpinConfiguration(s::UpdatingAlgorithm, spinConfiguration::AbstractVector{<:Number}) = (s.spinSystem.spinConfiguration = spinConfiguration)
getCouplingCoefficients(s::UpdatingAlgorithm)::AbstractMatrix{<:AbstractFloat} = s.spinSystem.couplingCoefficients
setCouplingCoefficients(s::UpdatingAlgorithm, couplingCoefficient::AbstractMatrix{<:AbstractFloat}) = (s.spinSystem.couplingCoefficients = couplingCoefficient)
getExternalMagneticField(s::UpdatingAlgorithm)::AbstractVector{<:AbstractFloat} = s.spinSystem.externalMagneticField
setExternalMagneticField(s::UpdatingAlgorithm, externalMagneticField::AbstractVector{<:AbstractFloat}) = (s.spinSystem.externalMagneticField = externalMagneticField)

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

################ For bipartite graphs ################

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
setSpinConfiguration(s::UpdatingAlgorithmOnBipartiteGraph, spinConfiguration::AbstractVector{<:Number}) = (s.spinSystem.spinConfiguration = spinConfiguration)
getHiddenLayer(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractVector{<:Number} = s.spinSystem.hiddenLayer
setHiddenLayer(s::UpdatingAlgorithmOnBipartiteGraph, hiddenLayer::AbstractVector{<:Number}) = (s.spinSystem.hiddenLayer = hiddenLayer)
getCouplingCoefficients(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractMatrix{<:AbstractFloat} = s.spinSystem.couplingCoefficients
setCouplingCoefficients(s::UpdatingAlgorithmOnBipartiteGraph, couplingCoefficients::AbstractMatrix{<:AbstractFloat}) = (s.spinSystem.couplingCoefficients = couplingCoefficients)
getExternalMagneticField(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractVector{<:AbstractFloat} = s.spinSystem.externalMagneticField
setExternalMagneticField(s::UpdatingAlgorithmOnBipartiteGraph, externalMagneticField::AbstractVector{<:AbstractFloat}) = (s.spinSystem.externalMagneticField = externalMagneticField)
getAuxiliaryBias(s::UpdatingAlgorithmOnBipartiteGraph)::AbstractVector{<:AbstractFloat} = s.spinSystem.auxiliaryBias
setAuxiliaryBias(s::UpdatingAlgorithmOnBipartiteGraph, auxiliaryBias::AbstractVector{<:AbstractFloat}) = (s.spinSystem.auxiliaryBias = auxiliaryBias)

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

end
