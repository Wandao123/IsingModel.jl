using IsingModel
using Test

@testset "IsingModel.jl" begin
    s = SpinSystem([-1, +1], [0 1; 1 0], [0, 0])
    println(calcEnergy(s))
    println(IsingModel.calcLocalMagneticField(s))
    a = AsynchronousHopfieldNetwork(deepcopy(s))
    update!(a)
    a = GlauberDynamics(deepcopy(s), 10.0)
    update!(a)
    a = MetropolisMethod(deepcopy(s), 10.0)
    update!(a)
end
