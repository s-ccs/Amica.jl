using Amica
using Test

x = reshape(reinterpret(Float32, read("test/tmpdata81472/tmpdata81472.fdt")), (32, :))

amica = fit(SingleModelAmica, x;
    lrate=Amica.LearningRate(;
        # lrate 0.050000
        lrate=0.05,
        # minlrate 1.00000e-08
        minimum=1.0e-08,
        # lratefact 0.500000
        decreaseFactor=0.5
    ),
    shapelrate=Amica.LearningRate(;
        # ? rholrate 0.050000
        lrate=0.05,
        # minrho 1.00000
        minimum=1,
        # maxrho 2.00000
        maximum=2,
        # ? rho0 1.50000
        init=1.5,
        # rholratefact 0.500000
        decreaseFactor=0.5
    ),
    # do_mean 1
    remove_mean=true,
    # do_sphere 1
    do_sphering=true,
    show_progress=true,
    # max_iter 10
    maxiter=10, do_newton=1,
    # newt_ramp 10
    newt_start_iter=10,
    iterwin=10,
    update_shape=1,
    # min_dll 1.000000e-09
    mindll=1.00000e-09,
)

@test amica.A ≈ reshape(reinterpret(Float64, read("test/tmpdata81472/amicaout/A")), size(amica.A))
@test amica.LL ≈ reshape(reinterpret(Float64, read("test/tmpdata81472/amicaout/LL")), size(amica.LL))
# This is not working yet due to different shapes :o
@test amica.Lt ≈ reshape(reinterpret(Float64, read("test/tmpdata81472/amicaout/LLt")), size(amica.Lt))
