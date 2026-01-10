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


0. großes datensatz
-> 190ms/iter, 24gb mem
1. newton
-> done
2. memory nsys analysieren
3. stride


issues:
- cpu langsam
- 

gute version:



## neue todos
- großen datensatz mit fortran-code testen
- fp & zfp in funktion?
- log: fastmath oä?
- benchmarken multithreaded julia vs mkl
- blocks einführen?
- verschiedene Versionen niederschreiben

## Thesis
https://arxiv.org/pdf/2506.10156
- Vergleich 32bit vs 64bit?
- Welche Parameter könnte man tunen? m?
- Runtime + Memory vergleich, Stability


# was man später noch machen kann
- auf random sample lernen statt ganzen datensatz