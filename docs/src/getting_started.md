```@meta
CurrentModule = Amica
```

# Getting Started

This tutorial will show how to build a synthetic dataset of mixtures, fit an AMICA model to it and recover the sources using the fitted model.

## Creating a Synthetic Dataset

We create three source signals: `sin.(t)`, `sign.(sin.(0.5 .* t .+ 0.3))` and a random source `randn(N)` and mix them using the `mixing` matrix.

```@example synthetic_workflow
using Amica
using CairoMakie # hide
using Random

Random.seed!(2)

N = 4_000
t = range(0, 20pi, length=N)

sources = hcat(
    sin.(t),
    sign.(sin.(0.5 .* t .+ 0.3)),
    randn(N),
)

mixing = [
    1.0 0.40 -0.25
    -0.35 1.10 0.30
    0.20 -0.45 1.30
]

mixtures = sources * mixing'
nothing # hide
```

The original source signals and the mixed signals look like this:

```@setup synthetic_workflow
function save_signal_plot(data, title, label_prefix, filename)
    fig = Figure(size=(900, 500))
    colors = [:tomato, :steelblue, :darkgreen]
    axes_list = [
        Axis(fig[i, 1]; ylabel="$(label_prefix) $i", title=i == 1 ? title : "") for i in axes(data, 2)
    ]

    for i in axes(data, 2)
        lines!(axes_list[i], 1:size(data, 1), data[:, i]; color=colors[i])
    end

    linkxaxes!(axes_list...)
    hidexdecorations!(axes_list[1]; grid=false)
    hidexdecorations!(axes_list[2]; grid=false)
    axes_list[end].xlabel = "Sample"
    save(filename, fig)
    return nothing
end

function save_sources_and_mixtures_plot(sources, mixtures, filename)
    fig = Figure(size=(1000, 420))
    colors = [:tomato, :steelblue, :darkgreen]

    source_axes = [
        Axis(fig[i, 1]; ylabel="Source $i", title=i == 1 ? "Source signals" : "") for i in axes(sources, 2)
    ]
    mixture_axes = [
        Axis(fig[i, 2]; ylabel="Mix $i", title=i == 1 ? "Mixed signals" : "") for i in axes(mixtures, 2)
    ]

    for i in axes(sources, 2)
        lines!(source_axes[i], 1:size(sources, 1), sources[:, i]; color=colors[i])
        lines!(mixture_axes[i], 1:size(mixtures, 1), mixtures[:, i]; color=colors[i])
    end

    linkxaxes!(source_axes...)
    linkxaxes!(mixture_axes...)

    hidexdecorations!(source_axes[1]; grid=false)
    hidexdecorations!(source_axes[2]; grid=false)
    hidexdecorations!(mixture_axes[1]; grid=false)
    hidexdecorations!(mixture_axes[2]; grid=false)

    source_axes[end].xlabel = "Sample"
    mixture_axes[end].xlabel = "Sample"

    save(filename, fig)
    return nothing
end

save_sources_and_mixtures_plot(sources, mixtures, "synthetic_sources_and_mixtures.svg")
nothing
```

![plot of mixed and unmixed sources](synthetic_sources_and_mixtures.svg)

## Fitting AMICA

The simplest way fit an AMICA model to the dataset we created, is to call the [`fit`](@ref) function.

```@example synthetic_workflow
model = fit(
    SingleModelAmica,
    mixtures;
    m=3,
    maxiter=50
)
```

## Using the Result

After fitting, `model` stores the learned parameters.

The most relevant fields are:

- `model.A`: learned mixing matrix, inverse of the unmixing matrix
- `model.LL`: log-likelihood for each iteration
- `model.proportions`, `model.location`, `model.scale`, `model.shape`: source density parameters of the fitted Gaussian Mixture Models

The [`recover_sources`](@ref) function can be used to unmix the data according to `model.A`.

```@example synthetic_workflow
recovered_sources = recover_sources(mixtures, model)

save_signal_plot(recovered_sources, "Recovered signals", "Recovered", "recovered_sources.svg") # hide
nothing # hide
```

A plot of the unmixed sources looks as follows and clearly shows that the original sources were successfully recovered, even though they might differ in order, sign and scale.

![plot of unmixed sources](recovered_sources.svg)
