# small.fdt

Matthew Barras and Liam Booth (2026). Cognitive Workload 8-level arithmetic. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds007262.v1.0.2

Converted using

```jl
using Statistics

using PyMNE

vhdr = "<path>/sub-001_task-arithmetic_eeg.vhdr"

raw = PyMNE.io.read_raw_brainvision(vhdr; preload=true)
raw.filter(l_freq=1.0, h_freq=nothing)   # high-pass at 1 Hz
data = Float32.(pyconvert(Array, raw.get_data(picks="eeg")))

write("small.fdt", data)
```

# Memorize.fdt

71 channel Sternberg dataset from Onton and Makeig 2006
Obtained from https://sccn.ucsd.edu/~jason/amica_web.html
