# Amica

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://s-ccs.github.io/Amica.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://s-ccs.github.io/Amica.jl/dev)
[![Test workflow status](https://github.com/s-ccs/Amica.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/s-ccs/Amica.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/s-ccs/Amica.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/s-ccs/Amica.jl)
[![Lint workflow Status](https://github.com/s-ccs/Amica.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/s-ccs/Amica.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/s-ccs/Amica.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/s-ccs/Amica.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/s-ccs/Amica.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

## AMICA.jl
Adaptive Independent Component Analysis on GPU in pure Julia.

## Speed
- Single CPU performance is slightly faster than the fortran implementation
- GPU is fastest
- 64 core multi-threading CPU is faster for Fortran, this difference appears at utilizing ~8 cores in our testing
<img width="700" height="320" alt="grafik" src="https://github.com/user-attachments/assets/2ead53ff-d2c9-46f0-bbb6-bde9e789582e" />

## Correctness
We checked our implementation against the Fortran implementation from Jason Palmer. This check is also implemented as a continuous integration check for future versions.
- Float32 did not impact performance in our three tested datasets
<img width="700" height="390" alt="grafik" src="https://github.com/user-attachments/assets/26327794-9fa8-4810-9b53-3c80692b8c14" />

## GPU Support
In theory, GPU support should work on Nvidia, AMD, Apple-Metal, and Intel as well. In practice, we only could test Nvidia and Apple-Metal so far. The support hinges on KernelAbstractions.jl to support more backends. Please raise an issue if you run into problems, we can surely figure it out!

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://s-ccs.github.io/Amica.jl/dev/90-contributing/)

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/v-morlock"><img src="https://avatars.githubusercontent.com/u/18666999?v=4?s=100" width="100px;" alt="Valentin Morlock"/><br /><sub><b>Valentin Morlock</b></sub></a><br /><a href="#code-v-morlock" title="Code">💻</a> <a href="#doc-v-morlock" title="Documentation">📖</a> <a href="#bug-v-morlock" title="Bug reports">🐛</a> <a href="#infra-v-morlock" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.benediktehinger.de"><img src="https://avatars.githubusercontent.com/u/10183650?v=4?s=100" width="100px;" alt="Benedikt Ehinger"/><br /><sub><b>Benedikt Ehinger</b></sub></a><br /><a href="#code-behinger" title="Code">💻</a> <a href="#infra-behinger" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AlexLulkin"><img src="https://avatars.githubusercontent.com/u/46380647?v=4?s=100" width="100px;" alt="AlexLulkin"/><br /><sub><b>AlexLulkin</b></sub></a><br /><a href="#code-AlexLulkin" title="Code">💻</a> <a href="#infra-AlexLulkin" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
