<!-- - LL plotten
- newton_method vergleichen
- wo wird location berechnet wenn m < 1 ?
- strided ausprobieren
- digamma -> \_evalpoly

neue logik cpu:
julia> @btime Amica.update_parameters!(myAmica, lrate, true, true)
1.654 s (167 allocations: 66.95 KiB)

loops logik cpu:
julia> @btime Amica.update_parameters!(myAmica, lrate, true)
680.697 ms (0 allocations: 0 bytes)

neue logik gpu:
julia> @btime Metal.@sync Amica.update_parameters!(myAmica, lrate, true, true);
377.276 ms (24881 allocations: 587.52 KiB)

broadcasts gpu: -->

- gpu profilen
- auf server probieren
- advanced cuda anschauen
