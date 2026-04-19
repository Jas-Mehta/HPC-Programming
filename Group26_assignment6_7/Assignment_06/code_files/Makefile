# ==========================================
# Makefile for Assignment 6: Parallel Interpolation
# ==========================================
# For HPC Cluster (Linux/GCC): make
# For macOS (clang + Homebrew libomp): make mac

CXX = g++
CXXFLAGS = -O3 -fopenmp -march=native -ffast-math -funroll-loops
LDFLAGS = -fopenmp -lm

SRCS = main.cpp utils.cpp init.cpp
TARGET = parallel.out
INPUT_MAKER = input_maker.out

all: $(TARGET) $(INPUT_MAKER)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(INPUT_MAKER): input_file_maker.cpp
	$(CXX) -O2 -o $@ $^

# macOS target (clang++ with Homebrew libomp)
mac:
	clang++ -O3 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -o $(TARGET) $(SRCS) -L/opt/homebrew/opt/libomp/lib -lomp -lm
	clang++ -O2 -o $(INPUT_MAKER) input_file_maker.cpp

clean:
	rm -f $(TARGET) $(INPUT_MAKER) Mesh.out input.bin

.PHONY: all clean mac
