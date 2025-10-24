| var            | Bedeutung                                                   | fortran  |     |     |
| -------------- | ----------------------------------------------------------- | -------- | --- | --- |
| M              | Number of ICA Models                                        |          |     |     |
| h              | one ICA model                                               |          |     |     |
| m              | number of source densitiy mixtures (= 3 )                   |          |     |     |
| N              | nSamples                                                    |          |     |     |
| n              | number of channels == components                            |          |     |     |
| x              | data (n,N)                                                  |          |     |     |
| centers        | model centers                                               | c        |     |     |
| ?              |                                                             | gm       |     |     |
| alpha          | mixture proportion => prior mixture probabilities? => $\pi$ |          |     |     |
| mu             | location of mixtures => $\mu$                               | location |     |     |
| sbeta          | scale of mixtures $\alpha$                                  | scale    |     |     |
| rho            | shape of mixtures => $\rho$                                 |          |     |     |
| b / s          | source activation s / b                                     |          |     |     |
| A              | mixing matrix -> $s*A =x$                                   | A        |     |     |
| W              | unmixing matrix $x*W = s$ $W =A^{-1}$                       | W        |     |     |
| z              | difference of likelihoods between mixtures?                 |          |     |     |
| y              | source signals                                              | y        |     |     |
| source_signals | unmixed signals                                             | b        |     |     |
| u              |                                                             |          |     |     |
| Q              | log likelihood of individual mixtures `m` over time         | Q        |     |     |
| Lt             | log Likelihood                                              | Ptmp     |     |     |
| v              |                                                             |          |     |     |
| dLL            | difference log likelihood                                   |          |     |     |
| iterwin        | moving averaging                                            |          |     |     |
| dA             | delta change for mixing matrix                              |          |     |     |
| g              | gradient                                                    |          |     |     |
|                |                                                             |          | eta |     |
| kappa          | hessian?                                                    |          |     |     |
| sigma2         | hessian?                                                    |          |     |     |
| gm             |                                                             |          |     |     |
| v              |                                                             |          |     |     |
|                |                                                             |          |     |     |
| iter           | current iteration                                           |          |     |     |
| maxiter        | 2000 default                                                |          |     |     |
| fix_init       | 1 (randomize init?)                                         |          |     |     |
| ldet           | determinant?                                                | DSum     |     |     |
|                | weighted centers                                            | wc       |     |     |
