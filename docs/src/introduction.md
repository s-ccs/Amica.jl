# Introduction

## Data Shape

`Amica.jl` expects a matrix with this shape:

- rows: samples / time points
- columns: channels / observed mixtures

So for a dataset with `N` samples and `n` channels, its size should be `(N, n)`.

```julia
N = 10_000
n = 32
data = rand(Float64, N, n)
```

## Minimum Example

This initializes a random matrix and fits an AMICA model with 3 components over 10 iterations. It uses the (currently only) AMICA implementation `SingleModelAmica`.

```@example
using Amica

data = rand(Float64, 10_000, 32)

model = fit(
    SingleModelAmica,
    data;
    m=3,
    maxiter=10,
)
```
