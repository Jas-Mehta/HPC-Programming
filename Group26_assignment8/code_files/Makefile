# ============================================================
# Makefile — Assignment 08: MPI + OpenMP Particle Interpolation
# ============================================================
# Targets:
#   make              → build parallel.out + input_maker.out
#   make verify       → correctness test against Test_Mesh.out
#   make configs      → generate all 5 benchmark input files
#   make clean        → remove build artefacts

CXX      = mpicxx
CXXFLAGS = -O3 -fopenmp -march=native -ffast-math -funroll-loops
LDFLAGS  = -fopenmp -lm

SRCS   = main.cpp utils.cpp
TARGET = parallel.out
IMAKER = input_maker.out

# ---- Default build ---------------------------------------------------
all: $(TARGET) $(IMAKER)

$(TARGET): $(SRCS) init.h utils.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

$(IMAKER): input_file_maker.cpp
	$(CXX) -O2 -o $@ $^

# ---- Correctness verification ----------------------------------------
# Runs serial (1 rank, 1 thread), then compares Mesh.out to Test_Mesh.out.
verify: $(TARGET)
	OMP_NUM_THREADS=1 mpirun -np 1 ./$(TARGET) Test_input.bin
	@awk 'BEGIN{maxe=0} \
	    NR==FNR{n=split($$0,a); for(i=1;i<=n;i++) ref[FNR,i]=a[i]; rows=FNR; cols=n; next} \
	    {n=split($$0,a); for(i=1;i<=n;i++){d=ref[FNR,i]-a[i]; if(d<0)d=-d; if(d>maxe)maxe=d}} \
	    END{printf "Max abs error: %.2e\n",maxe; if(maxe>1e-4){print "FAIL"; exit 1} print "PASS"}' \
	    Test_Mesh.out Mesh.out

# ---- Generate the 5 benchmark input configurations -------------------
# (a) Nx=250,  Ny=100,  0.9 M  particles, 10 iters
# (b) Nx=250,  Ny=100,  5   M  particles, 10 iters
# (c) Nx=500,  Ny=200,  3.6 M  particles, 10 iters
# (d) Nx=500,  Ny=200,  20  M  particles, 10 iters
# (e) Nx=1000, Ny=400,  14  M  particles, 10 iters
configs: $(IMAKER)
	./$(IMAKER) 250  100   900000 10 config_a.bin
	./$(IMAKER) 250  100  5000000 10 config_b.bin
	./$(IMAKER) 500  200  3600000 10 config_c.bin
	./$(IMAKER) 500  200 20000000 10 config_d.bin
	./$(IMAKER) 1000 400 14000000 10 config_e.bin

# ---- Cleanup ---------------------------------------------------------
clean:
	rm -f $(TARGET) $(IMAKER) Mesh.out results.csv \
	      config_a.bin config_b.bin config_c.bin config_d.bin config_e.bin

.PHONY: all verify configs clean
