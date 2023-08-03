using IsingModel
using Test

@testset "IsingModel.jl" begin
    s = SpinSystem([-1, +1], [0 1; 1 0], [0, 0])
    println("Energy = $(calcEnergy(s))")
    a = AsynchronousHopfieldNetwork(deepcopy(s))
    update!(a)
    a = GlauberDynamics(deepcopy(s), 10.0)
    update!(a)
    a = MetropolisMethod(deepcopy(s), 10.0)
    update!(a)
    for (i, sample) in zip(0:10, takeSamples!(a, 10, n -> a.temperature))
        println("$i: $sample")
    end
end
