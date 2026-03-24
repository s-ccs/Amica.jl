# Configuration

## Common Options

- `m`: number of mixture components per source, default `3`
- `maxiter`: maximum number of iterations
- `block_size`: number of samples per block
- `num_threads`: number of CPU threads
- `remove_mean`: subtract the column-wise mean before fitting
- `do_sphering`: sphere the data before fitting
- `show_progress`: print progress during fitting

## Providing Initial Parameters

Instead letting the algorithm initialize them, certain initial values can be passed:

```julia
N, n = size(data)
m = 3

A = Matrix{Float64}(I, n, n)
location = zeros(Float64, n, m)
scale = ones(Float64, n, m)

model = SingleModelAmica(
    Float64;
    nsamples=N,
    ncomps=n,
    m=m,
    A=A,
    location=location,
    scale=scale,
)
```

## GPU Arrays

For an additional speedup, `Amica.jl` can work with other array backends through `ArrayType`, but only CUDA is thoroughly tested at the moment.

```julia
using Amica
using CUDA

data = rand(Float32, 10_000, 32)

model = fit(
    SingleModelAmica,
    data;
    ArrayType=CuArray,
    m=3,
    maxiter=50,
    block_size=10_000,
)
```
