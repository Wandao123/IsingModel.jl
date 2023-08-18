module IsingModel

export SpinSystems
export SingleSpinFlip
export MultiSpinFlip
export OnBipartiteGraph
export SamplingHelper

@enum IsingSpin DownSpin = -1 UpSpin = +1

include("SpinSystems.jl")
include("SingleSpinFlip.jl")
include("MultiSpinFlip.jl")
include("OnBipartiteGraph.jl")
include("SamplingHelper.jl")

end
