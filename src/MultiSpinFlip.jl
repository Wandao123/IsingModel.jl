module MultiSpinFlip

export update!, makeSampler!

using LinearAlgebra
using Random, Distributions
using ..SpinSystems

abstract type MultiSpinUpdatingAlgorithm <: UpdatingAlgorithm end

end
