using IsingModel
using SparseArrays
using Test

@testset "IsingModel.jl" begin
    s = SpinSystem([-1, +1], sparse([0 1; 1 0]), [0, 0])
    println("Energy = $(calcEnergy(s))")
    a = AsynchronousHopfieldNetwork(deepcopy(s))
    for (i, sample) in zip(0:10, takeSamples!(a, 5))
        println("$i: $sample")
    end
    a = GlauberDynamics(deepcopy(s), 10.0)
    for (i, sample) in zip(0:10, takeSamples!(a, 5, n -> 10.0^(-n)))
        println("$i: $sample")
    end
    a = MetropolisMethod(deepcopy(s), 10.0)
    for (i, sample) in zip(0:10, takeSamples!(a, 5, n -> 10.0 ^ (-n)))
        println("$i: $sample")
    end
end
