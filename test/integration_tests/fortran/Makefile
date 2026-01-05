# Simple Makefile for mean_test (OpenMP + MPI)

FC      = mpif90
FFLAGS  = -O2 -w -cpp -fopenmp -ffree-line-length-none
# Use one BLAS: either Accelerate (macOS) or OpenBLAS (brew)
#LDFLAGS = -fopenmp -framework Accelerate
LDFLAGS = -fopenmp -L/opt/homebrew/opt/openblas/lib -lopenblas

all: amica

amica: funmod2.f90 amica17.f90
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f amica *.o *.mod

.PHONY: all clean
